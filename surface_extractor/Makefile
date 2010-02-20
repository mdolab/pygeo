default:
	ifort -c -O2 -fPIC surface_extractor.f90
	ifort -lcgns surface_extractor.o -o surface_extractor
clean:
	-rm surface_extractor
	-rm surface_extractor.o