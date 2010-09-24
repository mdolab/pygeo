default: intel

intel:
	ifort -c -O2 -fPIC surface_extractor.f90
	ifort -lcgns surface_extractor.o -o surface_extractor

gfortran: 
	gfortran -c -I/usr/local/include/ -O2 -fPIC surface_extractor.f90
	gfortran -lcgns surface_extractor.o -o surface_extractor
clean:
	-rm surface_extractor
	-rm surface_extractor.o