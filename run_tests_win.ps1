docker run -v ${pwd}:/root/test/pygeo --rm benjaminbrelje/mdolabfull:latest /bin/sh -c ". /root/.bashrc_mdolab && cd /root/test && python -m pytest pygeo/tests/reg_tests"
