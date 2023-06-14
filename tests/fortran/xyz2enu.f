        subroutine xyz2enu (la,lg,x,e)
            implicit none

            double precision la,lg,pi,dtr,x(3,1),e(3,1)
            double precision ca,sa,cg,sg,dx,dy,dz,de,dn,du

            pi = 4.d0*datan(1.d0)
            dtr = pi/180.d0

c           Convert to radians from decimal degrees
            la = la*dtr
            lg = lg*dtr

            ca = dcos(la)
            sa = dsin(la)

            cg = dcos(lg)
            sg = dsin(lg)

            dx = x(1,1)
            dy = x(2,1)
            dz = x(3,1)

            dn = -sa*cg*dx -sa*sg*dy + ca*dz
            de = -   sg*dx +   cg*dy
            du =  ca*cg*dx +ca*sg*dy + sa*dz

            e(1,1) = de
            e(2,1) = dn
            e(3,1) = du

Cf2py intent(out) e

            return
        end


        subroutine enu2xyz (la,lg,e,x)

            implicit none

            double precision la,lg,pi,dtr,x(3,1),e(3,1)
            double precision ca,sa,cg,sg,dx,dy,dz,de,dn,du

            pi = 4.d0*datan(1.d0)
            dtr = pi/180.d0

            ca = dcos(la)
            sa = dsin(la)

            cg = dcos(lg)
            sg = dsin(lg)

            de = e(1,1)
            dn = e(2,1)
            du = e(3,1)

            dx = -sa*cg*dn  - sg*de  + ca*cg*du
            dy = -sa*sg*dn  + cg*de  + ca*sg*du
            dz =  ca   *dn  + 0      +   sa *du

            x(1,1) = dx
            x(2,1) = dy
            x(3,1) = dz

Cf2py intent(out) x

            return
        end
