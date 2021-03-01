import unittest
from docutils import nodes
from docutils.parsers.rst import Directive
import re
from sphinx.errors import SphinxError
import sphinx
import traceback
import inspect
import os
from io import StringIO
import tokenize 

from docutils.parsers.rst.directives import unchanged, images

from openmdao.docs._utils.docutil import get_source_code, remove_docstrings, \
    remove_initial_empty_lines, replace_asserts_with_prints, \
    strip_header, dedent, insert_output_start_stop_indicators, run_code, \
    get_skip_output_node, get_interleaved_io_nodes, get_output_block_node, \
    split_source_into_input_blocks, extract_output_blocks, consolidate_input_blocks, node_setup, \
    strip_decorators, remove_leading_trailing_whitespace_lines

def remove_excerpt_tags(source):
    """
    Return 'source' minus script excerpt tags.
    Parameters
    ----------
    source : str
        Original source code.
    Returns
    -------
    str
        Source with script excerpt tags removed.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    pattern = re.compile(r"# EXCERPT [0-9]+ #")

    prev_token_was_excerpt = False
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4] # in original code but not used here
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # This series of conditionals removes excerpt tags:
        if token_type == tokenize.COMMENT and pattern.match(token_string):
            prev_token_was_excerpt = True
        elif token_type == tokenize.NL and prev_token_was_excerpt:
            # kill newlines immediately after token comments
            prev_token_was_excerpt = False
        else:
            out += token_string
            prev_token_was_excerpt = False
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out

# This get_source_code directive is customized
def get_script_excerpt(path):
    """
    Return source code as a text string.

    Parameters
    ----------
    path : str
        Path to a file, module, function, class, or class method.

    Returns
    -------
    str
        The source code.
    int
        Indentation level.
    module or None
        The imported module.
    class or None
        The class specified by path.
    method or None
        The class method specified by path.
    """

    indent = 0
    class_obj = None
    method_obj = None

    split_path = path.split(':EXCERPT:')
    if len(split_path) != 2:
        raise SphinxError("Too many or too few excerpt numbers in '%s'" % path)
    excerpt_number = int(split_path[-1])
    path = split_path[0]
    if not path.endswith('.py'):
        raise SphinxError('Script excerpt tags only work with script files ending in .py')
    if not os.path.isfile(path):
        raise SphinxError("Can't find file '%s' cwd='%s'" % (path, os.getcwd()))
    with open(path, 'r') as f:
        source = f.read()
    module = None

    split_comment = '# EXCERPT ' + str(excerpt_number) + ' #'
    split_source = source.split(split_comment)
    if len(split_source) != 3:
        raise SphinxError("Too few or too many excerpt comment tags " + split_comment)
    else:
        source = split_source[1]
    
    return remove_leading_trailing_whitespace_lines(source), indent, module, class_obj, method_obj

_plot_count = 0

plotting_functions = ['\.show\(', 'partial_deriv_plot\(']


class EmbedCodeDirective(Directive):
    """
    EmbedCodeDirective is a custom directive to allow blocks of
    python code to be shown in feature docs.  An example usage would look like this:

    .. embed-code::
        openmdao.test.whatever.method

    What the above will do is replace the directive and its args with the block of code
    for the class or method desired.

    By default, docstrings will be kept in the embedded code. There is an option
    to the directive to strip the docstrings:

    .. embed-code::
        openmdao.test.whatever.method
        :strip-docstrings:
    """

    # must have at least one directive for this to work
    required_arguments = 1
    has_content = True

    option_spec = {
        'strip-docstrings': unchanged,
        'strip-excerpt-tags': unchanged,
        'layout': unchanged,
        'scale': unchanged,
        'align': unchanged,
        'imports-not-required': unchanged,
    }

    def run(self):
        global _plot_count

        #
        # error checking
        #
        allowed_layouts = set(['code', 'output', 'interleave', 'plot'])

        if 'layout' in self.options:
            layout = [s.strip() for s in self.options['layout'].split(',')]
        else:
            layout = ['code']

        if len(layout) > len(set(layout)):
            raise SphinxError("No duplicate layout entries allowed.")

        bad = [n for n in layout if n not in allowed_layouts]
        if bad:
            raise SphinxError("The following layout options are invalid: %s" % bad)

        if 'interleave' in layout and ('code' in layout or 'output' in layout):
            raise SphinxError("The interleave option is mutually exclusive to the code "
                              "and output options.")

        #
        # Get the source code
        #
        path = self.arguments[0]

                
        try:
            if len(path.split(':EXCERPT:')) > 1:
                source, indent, module, class_, method = get_script_excerpt(path)
            else:
                source, indent, module, class_, method = get_source_code(path)
        except Exception as err:
            # Generally means the source couldn't be inspected or imported.
            # Raise as a Directive warning (level 2 in docutils).
            # This way, the sphinx build does not terminate if, for example, you are building on
            # an environment where mpi or pyoptsparse are missing.
            raise self.directive_error(2, str(err))

        #
        # script, test and/or plot?
        #
        is_script = path.endswith('.py')

        is_test = class_ is not None and inspect.isclass(class_) and issubclass(class_, unittest.TestCase)

        shows_plot = re.compile('|'.join(plotting_functions)).search(source)

        if 'plot' in layout:
            plot_dir = os.getcwd()
            plot_fname = 'doc_plot_%d.png' % _plot_count
            _plot_count += 1

            plot_file_abs = os.path.join(os.path.abspath(plot_dir), plot_fname)
            if os.path.isfile(plot_file_abs):
                # remove any existing plot file
                os.remove(plot_file_abs)

        #
        # Modify the source prior to running
        #
        if 'strip-docstrings' in self.options:
            source = remove_docstrings(source)
        if 'strip-excerpt-tags' in self.options:
            source = remove_excerpt_tags(source)
        if is_test:
            try:
                source = dedent(source)
                source = strip_decorators(source)
                source = strip_header(source)
                source = dedent(source)
                source = replace_asserts_with_prints(source)
                source = remove_initial_empty_lines(source)

                class_name = class_.__name__
                method_name = path.rsplit('.', 1)[1]

                # make 'self' available to test code (as an instance of the test case)
                self_code = "from %s import %s\nself = %s('%s')\n" % \
                            (module.__name__, class_name, class_name, method_name)

                # get setUp and tearDown but don't duplicate if it is the method being tested
                setup_code = '' if method_name == 'setUp' else dedent(strip_header(remove_docstrings(
                    inspect.getsource(getattr(class_, 'setUp')))))

                teardown_code = '' if method_name == 'tearDown' else dedent(strip_header(
                    remove_docstrings(inspect.getsource(getattr(class_, 'tearDown')))))

                # for interleaving, we need to mark input/output blocks
                if 'interleave' in layout:
                    interleaved = insert_output_start_stop_indicators(source)
                    code_to_run = '\n'.join([self_code, setup_code, interleaved, teardown_code]).strip()
                else:
                    code_to_run = '\n'.join([self_code, setup_code, source, teardown_code]).strip()
            except Exception:
                err = traceback.format_exc()
                raise SphinxError("Problem with embed of " + path + ": \n" + str(err))
        else:
            if indent > 0:
                source = dedent(source)
            if 'interleave' in layout:
                source = insert_output_start_stop_indicators(source)
            code_to_run = source[:]

        #
        # Run the code (if necessary)
        #
        skipped = failed = False

        if 'output' in layout or 'interleave' in layout or 'plot' in layout:

            imports_not_required = 'imports-not-required' in self.options

            if shows_plot:
                # import matplotlib AFTER __future__ (if it's there)
                mpl_import = "\nimport matplotlib\nmatplotlib.use('Agg')\n"
                idx = code_to_run.find("from __future__")
                idx = code_to_run.find('\n', idx) if idx >= 0 else 0
                code_to_run = code_to_run[:idx] + mpl_import + code_to_run[idx:]

                if 'plot' in layout:
                    code_to_run = code_to_run + ('\nmatplotlib.pyplot.savefig("%s")' % plot_file_abs)

            if is_test and getattr(method, '__unittest_skip__', False):
                skipped = True
                failed = False
                run_outputs = method.__unittest_skip_why__
            else:
                skipped, failed, run_outputs = run_code(code_to_run, path, module=module, cls=class_,
                                                        imports_not_required=imports_not_required,
                                                        shows_plot=shows_plot)

        #
        # Handle output
        #
        if failed:
            # Failed cases raised as a Directive warning (level 2 in docutils).
            # This way, the sphinx build does not terminate if, for example, you are building on
            # an environment where mpi or pyoptsparse are missing.
            raise self.directive_error(2, run_outputs)
        elif skipped:
            # issue a warning unless it's about missing SNOPT when building a Travis pull request
            PR = os.environ.get("TRAVIS_PULL_REQUEST")
            if not (PR and PR != "false" and "pyoptsparse is not providing SNOPT" in run_outputs):
                self.state_machine.reporter.warning(run_outputs)

            io_nodes = [get_skip_output_node(run_outputs)]

        else:
            if 'output' in layout:
                output_blocks = run_outputs if isinstance(run_outputs, list) else [run_outputs]

            elif 'interleave' in layout:
                if is_test:
                    start = len(self_code) + len(setup_code)
                    end = len(code_to_run) - len(teardown_code)
                    input_blocks = split_source_into_input_blocks(code_to_run[start:end])
                else:
                    input_blocks = split_source_into_input_blocks(code_to_run)

                output_blocks = extract_output_blocks(run_outputs)

                # Merge any input blocks for which there is no corresponding output
                # with subsequent input blocks that do have output
                input_blocks = consolidate_input_blocks(input_blocks, output_blocks)

            if 'plot' in layout:
                if not os.path.isfile(plot_file_abs):
                    raise SphinxError("Can't find plot file '%s'" % plot_file_abs)

                directive_dir = os.path.relpath(os.getcwd(),
                                                os.path.dirname(self.state.document.settings._source))
                # this filename must NOT contain an absolute path, else the Figure will not
                # be able to find the image file in the generated html dir.
                plot_file = os.path.join(directive_dir, plot_fname)

                # create plot node
                fig = images.Figure(self.name, [plot_file], self.options, self.content, self.lineno,
                                    self.content_offset, self.block_text, self.state,
                                    self.state_machine)
                plot_nodes = fig.run()

        #
        # create a list of document nodes to return based on layout
        #
        doc_nodes = []
        skip_fail_shown = False
        for opt in layout:
            if opt == 'code':
                # we want the body of code to be formatted and code highlighted
                body = nodes.literal_block(source, source)
                body['language'] = 'python'
                doc_nodes.append(body)
            elif skipped:
                if not skip_fail_shown:
                    body = nodes.literal_block(source, source)
                    body['language'] = 'python'
                    doc_nodes.append(body)
                    doc_nodes.extend(io_nodes)
                    skip_fail_shown = True
            else:
                if opt == 'interleave':
                    doc_nodes.extend(get_interleaved_io_nodes(input_blocks, output_blocks))
                elif opt == 'output':
                    doc_nodes.append(get_output_block_node(output_blocks))
                else:  # plot
                    doc_nodes.extend(plot_nodes)

        return doc_nodes


def setup(app):
    """add custom directive into Sphinx so that it is found during document parsing"""
    app.add_directive('embed-code', EmbedCodeDirective)
    node_setup(app)

    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}