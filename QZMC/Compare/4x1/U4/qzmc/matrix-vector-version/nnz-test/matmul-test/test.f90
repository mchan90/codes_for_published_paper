program test
        implicit none
        integer          :: i, j, n_matrix
        double precision :: start_time, end_time, zr, zi
        double complex   :: M(4900,4900), v(4900)
        ! test
        do i = 1, 4900
                call random_number(zr)
                call random_number(zi)
                v(i) = dcmplx(zr,zi)
        end do
        do i = 1, 4900
                do j = 1, 4900
                        call random_number(zr)
                        call random_number(zi)
                        M(i,j) = dcmplx(zr,zi)
                end do
        end do
        call CPU_TIME(start_time)
        n_matrix = 40
        do i = 1, n_matrix
                v = matmul(M,v)
        end do
        call CPU_TIME(end_time)
        write(6,*) 'elapsed_time',end_time-start_time
end program test
