subroutine colinearity(axis, origin, coords, n, x, nCon)
  implicit none
  real, dimension(3)::origin, axis
  real, dimension(n,3)::coords
  real, dimension(nCon)::x
  integer:: n, nCon

  real, dimension(n,3)::dirVec
  real, dimension(3)::resultDir
  integer:: i,j
  

  !Compute the direction from each point to the origin
  do i = 1,n
     do j = 1,3
        dirVec(i,j) = origin(j)-coords(i,j)
     end do
  end do

  ! compute the cross product with the desired axis. Cross product
  ! will be zero if the direction vector is the same as the axis

 !$AD II-LOOP
  do i=1,nCon
     resultDir = cross(axis,dirVec(i,:))
 
     x(i) = 0
     do j =1,3
        x(i) = x(i)+(resultDir(j))**2
     end do
     x(i) = sqrt(x(i))
  end do

end subroutine colinearity

subroutine planarity(axis, origin, p0, p1, p2, n, x, nCon)
  implicit none
  real, dimension(3)::origin, axis
  real, dimension(n,3)::p0,p1,p2
  real, dimension(nCon)::x
  integer:: n, nCon

  real, dimension(3*n,3)::allPoints,dist
  real, dimension(3*n)::scalarDist
  real :: tmp
  integer:: i, j

  ! copy data into all points array
  allPoints(1:n,:) = p0
  allPoints(n:2*n,:) = p1
  allPoints(2*n:3*n,:) = p2

  ! Compute the distance from the origin to each point
  x(1) = 0
  do i=1,n*3
     do j=1,3
        dist(i,j) = allPoints(i,j)-origin(j)
     end do

     !project it onto the axis
     !x(i) = dot_product(axis,dist(i,:))
     call dot(axis,dist(i,:),scalarDist(i))
     x(1) = x(1)+ scalarDist(i)**2!/(n*3) 
     
  end do
  x(1) = sqrt(x(1))

end subroutine planarity

 
FUNCTION cross(a, b)
  real, DIMENSION(3) :: cross
  real, DIMENSION(3), INTENT(IN) :: a, b

  cross(1) = a(2) * b(3) - a(3) * b(2)
  cross(2) = a(3) * b(1) - a(1) * b(3)
  cross(3) = a(1) * b(2) - a(2) * b(1)
END FUNCTION cross

subroutine computeArea(p0, p1, p2, n, area) 
    real, dimension (n,3):: p0, p1, p2, v1, v2
    real, dimension (n,3):: crosses
    real, dimension (n):: areas
    real, INTENT(OUT) :: area 
    integer :: n 
    v1 = 0.
    v2 = 0.
    crosses = 0.
    areas = 0.
    !$AD II-LOOP
    do i=1,n
        do j = 1,3
            v1(i,j) = p1(i,j) - p0(i,j)
            v2(i,j) = p2(i,j) - p0(i,j)
        end do  
        crosses(i,:) = cross(v1(i,:),v2(i,:))
        do j = 1,3
            areas(i) = areas(i) + crosses(i,j)**2
        end do  
        areas(i) = sqrt(areas(i))
    end do 
    area = sum(areas)/2.
end subroutine

subroutine normalLengths(center, coords,n, X, nCon)

real, dimension(3)::center
real, dimension(n,3)::coords
real, dimension(nCon)::X
integer:: n,nCon

real ::refLength2,length2

refLength2 = 0
do i =1,3
 refLength2 = refLength2+(center(i)-coords(1,i))**2
end do

!$AD II-LOOP
do i =1,nCon
   length2 = 0
   do j =1,3
      Length2 = length2+(center(j)-coords(i+1,j))**2
   end do
   x(i) = sqrt(length2/reflength2)
end do
end subroutine normalLengths

SUBROUTINE PROJAREA(n, p0, p1, p2, axis, area)

  IMPLICIT NONE
! Input
  INTEGER, INTENT(IN) :: n
  REAL*8, INTENT(IN) :: p0(n, 3), p1(n, 3), p2(n, 3), axis(3)
! Output
  REAL*8, INTENT(OUT) :: area
! Working
  INTEGER :: i
  REAL*8 :: v1(3), v2(3), savec(3), pa
  DO i=1,n
    v1 = p1(i, :) - p0(i, :)
    v2 = p2(i, :) - p0(i, :)
    !CALL CROSS(v1, v2, savec)
    savec = CROSS(v1, v2)
    CALL DOT(savec, axis, pa)
    IF (pa .GT. 0) THEN
      area = area + pa
    ELSE
      pa = 0
    END IF
  END DO
  area = area/2.0
  
END SUBROUTINE PROJAREA


SUBROUTINE DOT(a, b, c)

  IMPLICIT NONE
  REAL*8, INTENT(IN) :: a(3), b(3)
  REAL*8, INTENT(OUT) :: c

  c = a(1)*b(1) + a(2)*b(2) + a(3)*b(3)
  
END SUBROUTINE DOT

