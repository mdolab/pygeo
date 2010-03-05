program surface_extractor
  
  ! Note:
  ! Max number of surface points is 100,000. Increase if necessary (line ~21)
  ! Max nuumber of boundary surfaces is 1000. (line ~22)

  implicit none
  include 'cgnslib_f.h'

  integer Ndim, i,boco
  parameter (Ndim = 3)
  integer  CellDim, PhysDim
  integer ier, n, zonetype
  character*32 name, filename,output_filename
  integer cg, base, zone,j
  integer nbases, nzones, size(Ndim*3)
  character*32 basename, zonename,boconame
  integer nbocos,bocotype
  integer NormalIndex(3), NormalListFlag, ndataset,datatype
  integer ptset_type, npnts, pnts(6),nval
  double precision x(100000),y(10000),z(10000),data_double(6)
  integer wall_bcs(1000,8),bc_counter ! Max number of walls
  CHARACTER *100 BUFFER

  ! Data in wall_bcs is:
  !(i,1) -> Zone id
  !(i,2) -> Boundary Condition ID
  !(i,3) -> I Low index
  !(i,4) -> J Low index
  !(i,5) -> K Low index
  !(i,6) -> I High index
  !(i,7) -> J High index
  !(i,8) -> K High index
  
  bc_counter = 1
  N = IARGC ()
  if (N .ne. 2) then
     print *,'Error: extract_surfs must be called with two arguments:'
     print *,'extract_surfs cgns_file.cngs plot3d_file.xyz'
     stop
  end if
  CALL GETARG(1 , filename)
  CALL GETARG(2 , output_filename)
  call cg_open_f(filename, MODE_READ, cg, ier)
  if (ier .eq. ERROR) call cg_error_exit_f
  print *,'Reading input  file: ',filename

  call cg_nbases_f(cg, nbases, ier)
  if (ier .eq. ERROR) call cg_error_exit_f

  if (nbases .gt. 1) then
     print *, 'Error: This program only handes CGNS files with one base'
     stop
  end if

  base = 1

  call cg_base_read_f(cg, base, basename, CellDim, PhysDim, ier)
  if (ier .eq. ERROR) call cg_error_exit_f
  if (cellDim .ne. 3 .or. PhysDim .ne. 3) then
     print *,'The Cells must be hexahedreal in 3 dimensions'
     stop
  end if
  !       *** base attribute:  GOTO base node
  call cg_goto_f(cg, base, ier, 'end')
  if (ier .eq. ERROR) call cg_error_exit_f
  call cg_nzones_f(cg, base, nzones, ier)
  if (ier .eq. ERROR) call cg_error_exit_f

  do zone=1, nzones
     call cg_zone_read_f(cg, base, zone, zonename, size, ier)
     if (ier .eq. ERROR) call cg_error_exit_f
     call cg_zone_type_f(cg, base, zone, zonetype, ier)
     if (ier .eq. ERROR) call cg_error_exit_f

     !       Get Boundary conditions
     call cg_nbocos_f(cg, base, zone, nbocos, ier)
     if (ier .eq. ERROR) call cg_error_exit_f
     do boco=1, nbocos
           call cg_boco_info_f(cg, base, zone, boco, boconame,bocotype,&
             ptset_type,npnts,NormalIndex,NormalListFlag,datatype,ndataset,ier)
        if (ier .eq. ERROR) call cg_error_exit_f
        if (BCTypeName(bocotype) == 'BCWallViscous') then
           ! Now determine if we actually have a 2D face zone
           call cg_boco_read_f(cg, base, zone, boco, pnts,data_double, ier)
           if ( (pnts(4)-pnts(1) > 0 .and. pnts(5)-pnts(2) > 0 ) .or. &
                (pnts(4)-pnts(1) > 0 .and. pnts(6)-pnts(3) > 0 ) .or. &
                (pnts(5)-pnts(2) > 0 .and. pnts(6)-pnts(3) > 0 )) then
              
              wall_bcs(bc_counter,1) = zone
              wall_bcs(bc_counter,2) = boco
              wall_bcs(bc_counter,3:8) = pnts(1:6)
              bc_counter = bc_counter + 1
              if (.not.(npnts == 2)) then
                 print *,'Error: Detected a wall Boundary Condition, but more than dimensions associated with it'
                 stop
              end if
           end if
        end if
     end do
  end do
  ! We can now write the plot3d header
  print *,'Writing output file: ',output_filename
  OPEN (7, FILE = output_filename)
  ! Write out all the sizes
  write(7,*) bc_counter-1
  do i=1,bc_counter-1
     write(7,*) (wall_bcs(i,6)-wall_bcs(i,3)+1),(wall_bcs(i,7)-wall_bcs(i,4)+1), (wall_bcs(i,8)-wall_bcs(i,5)+1)
  end do
  ! Now we go back and just extract the data we need
  do i=1,bc_counter-1
     ! Go to the zone
     !   lower range index
     nval = (wall_bcs(i,6)-wall_bcs(i,3)+1)*(wall_bcs(i,7)-wall_bcs(i,4)+1)*(wall_bcs(i,8)-wall_bcs(i,5)+1)
     !   read grid coordinates
     call cg_coord_read_f(cg,base,wall_bcs(i,1),'CoordinateX',RealDouble,&
          wall_bcs(i,3:5),wall_bcs(i,6:8),x,ier)
     call cg_coord_read_f(cg,base,wall_bcs(i,1),'CoordinateY',RealDouble,&
          wall_bcs(i,3:5),wall_bcs(i,6:8),y,ier)
     call cg_coord_read_f(cg,base,wall_bcs(i,1),'CoordinateZ',RealDouble,&
          wall_bcs(i,3:5),wall_bcs(i,6:8),z,ier)
     ! Now just dump coordinates
     do j=1,nval
        write(7,*) x(j)
     end do
     do j=1,nval
        write(7,*) y(j)
     end do
     do j=1,nval
        write(7,*) z(j)
     end do
  end do
  close(7)         
  print *,'Processed:',bc_counter-1,'zones'
  call cg_close_f(cg, ier)
end program surface_extractor
