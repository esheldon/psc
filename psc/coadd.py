"""
"""

import ngmix
import numpy as np
import galsim
import copy

class Coadder():
    def __init__(self,
                 observations,
                 interp='lanczos3',
                 flat_wcs=False,
                 jacobian=None,
                 rng=None):
        """
        parameters
        -----------
        observations: ngmix.ObsList
            Set of observations to coadd. The observations must have
            an example noise image in a .noise attribute
        interp: string, optional
            Interpolation to use
        flat_wcs: bool
            If True, make the coadd have a flat wcs.  Default is
            to use the WCS of the first observation in the list
        rng: galsim random deviate, optional
            for creating the noise images 
        """
        self.observations = observations
        self.interp = interp
        self.flat_wcs=flat_wcs
        self.jacobian=jacobian
        self.rng=rng

        # use a nominal sky position
        self.sky_center = galsim.CelestialCoord(
            5*galsim.hours,
            -25*galsim.degrees,
        )

        self._process_obs()
        self._set_coadded_image()
        self._set_coadded_psf()
        self._set_coadded_weight_map()
        self._set_coadd_jacobian_cen()
        self._set_coadd_psf_jacobian_cen()
        self.coadd_obs.update_meta_data(self.observations.meta)

    def get_coadd(self):
        """
        Get the coadd observation; this returns a reference
        to the .coadd_obs attribute
        """

        return self.coadd_obs
    
    def _set_coadded_weight_map(self):
        """
        coadd the noise image realizations and take var

        also set the .noise attribute
        """
        coadd_obs=self.coadd_obs
        weights=self.weights

        weight_map = np.zeros( (self.ny, self.nx))

        wcs = coadd_obs.jacobian.get_galsim_wcs()

        coadd_noise = galsim.Sum([image*w for image,w in zip(self.noise_images,weights)])
        coadd_noise_image = galsim.Image(self.nx, self.ny, wcs=wcs)
        coadd_noise.drawImage(image=coadd_noise_image, method='no_pixel')

        weight_map[:,:] = 1./np.var(coadd_noise_image.array)
        coadd_obs.set_weight(weight_map)

        coadd_obs.noise = coadd_noise_image.array

    def _set_coadd_jacobian_cen(self):
        """
        set the center

        currently only support the canonical center
        """
        cen = self.canonical_center
        self.coadd_obs.jacobian.set_cen(
            row=cen.y-1,
            col=cen.x-1,
        )

    def _set_coadd_psf_jacobian_cen(self):
        """
        set the center

        currently only support the canonical center
        """
        cen = self.psf_canonical_center
        self.coadd_obs.psf.jacobian.set_cen(
            row=cen.y-1,
            col=cen.x-1,
        )

    def _set_coadded_image(self):
        """
        do the actual coadding, with appropriate weights

        wcs of final image is that of the *first*, since
        the coadd obs is a copy of that
        """
        coadd_obs=self.coadd_obs
        weights=self.weights

        wcs = coadd_obs.jacobian.get_galsim_wcs()

        coadd = galsim.Sum([image*w for image,w in zip(self.images,weights)])

        coadd_image = galsim.Image(self.nx, self.ny, wcs=wcs)
        coadd.drawImage(
            image=coadd_image,
            method='no_pixel',
        )

        coadd_obs.set_image(coadd_image.array)

    def _set_coadded_psf(self):
        """
        set the coadd psf

        wcs of final psf image is that of the *first*, since
        the coadd obs is a copy of that
        """
        coadd_obs=self.coadd_obs
        weights=self.weights

        psf_wcs = coadd_obs.psf.jacobian.get_galsim_wcs()

        coadd_psf = galsim.Sum([psf*w for psf,w in zip(self.psfs,weights)])
        coadd_psf_image = galsim.Image(self.psf_nx, self.psf_ny, wcs=psf_wcs)
        coadd_psf.drawImage(image=coadd_psf_image, method='no_pixel')

        coadd_psf_image = coadd_psf_image.array

        coadd_obs.psf.set_image(coadd_psf_image)

    def _set_coadd_obs_same(self):
        """
        base the wcs for the coadd off the first observation
        """

        for i,obs in enumerate(self.observations):
            if i==0:
                ny,nx = obs.image.shape
                pny,pnx = obs.psf.image.shape
            else:
                tny,tnx = obs.image.shape
                tpny,tpnx = obs.psf.image.shape

                if ny != tny or nx != tnx:
                    raise ValueError(
                        "ps sizes don't match: "
                        "[%d,%d] vs [%d,%d]" % (ny,nx,tny,tnx)
                    )
                if pny != tpny or pnx != tpnx:
                    raise ValueError(
                        "psf ps sizes don't match: "
                        "[%d,%d] vs [%d,%d]" % (pny,pnx,tpny,tpnx)
                    )

        tim = galsim.ImageD(nx,ny)
        self.canonical_center = tim.true_center

        ptim = galsim.ImageD(pnx,pny)
        self.psf_canonical_center = ptim.true_center

        self.nx=nx
        self.ny=ny
        self.psf_nx=pnx
        self.psf_ny=pny

        obs0 = self.observations[0]

        ojac = obs0.jacobian
        opjac = obs0.psf.jacobian

        if self.flat_wcs:
            jac = ngmix.DiagonalJacobian(
                row=ojac.get_row0(),
                col=ojac.get_col0(),
                scale=ojac.get_scale(),
            )
            pjac = ngmix.DiagonalJacobian(
                row=opjac.get_row0(),
                col=opjac.get_col0(),
                scale=opjac.get_scale(),
            )
        else:
            jac = ojac.copy()
            pjac = opjac.copy()

        psf_obs = ngmix.Observation(
            ptim.array,
            weight=ptim.array*0 + 1.0,
            jacobian=pjac,
        )

        self.coadd_obs = ngmix.Observation(
            tim.array,
            weight=tim.array*0 + 1.0,
            jacobian=jac,
            psf=psf_obs,
        )

    def _set_coadd_obs(self):
        """
        base the coadd off the observation with largest
        postage stamp

        But for consistency, we always take the jacobian
        from the first. This way we always know which
        wcs has been used
        """
        nxs=np.zeros(len(self.observations),dtype='i8')
        nys=nxs.copy()
        pnxs=nxs.copy()
        pnys=nxs.copy()

        for i,obs in enumerate(self.observations):
            ny,nx = obs.image.shape
            nxs[i] = nx
            nys[i] = ny

            pny,pnx = obs.psf.image.shape
            pnxs[i] = pnx
            pnys[i] = pny

        argx = nxs.argmax()
        argy = nys.argmax()
        pargx = pnxs.argmax()
        pargy = pnys.argmax()

        assert argx==argy

        nx = nxs[argx]
        ny = nys[argy]
        pnx = pnxs[pargx]
        pny = pnys[pargy]

        tim = galsim.ImageD(nx,ny)
        self.canonical_center = tim.true_center

        ptim = galsim.ImageD(pnx,pny)
        self.psf_canonical_center = ptim.true_center

        self.nx=nx
        self.ny=ny
        self.psf_nx=pnx
        self.psf_ny=pny


        if self.jacobian is not None:
            jac = self.jacobian.copy()
            pjac = self.jacobian.copy()
        else:

            obs0 = self.observations[0]
            ojac = obs0.jacobian
            opjac = obs0.psf.jacobian
            if self.flat_wcs:
                jac = ngmix.DiagonalJacobian(
                    row=ojac.get_row0(),
                    col=ojac.get_col0(),
                    scale=ojac.get_scale(),
                )
                pjac = ngmix.DiagonalJacobian(
                    row=opjac.get_row0(),
                    col=opjac.get_col0(),
                    scale=opjac.get_scale(),
                )
            else:
                jac = ojac.copy()
                pjac = opjac.copy()

        psf_obs = ngmix.Observation(
            ptim.array,
            weight=ptim.array*0 + 1.0,
            jacobian=pjac,
        )

        self.coadd_obs = ngmix.Observation(
            tim.array,
            weight=tim.array*0 + 1.0,
            jacobian=jac,
            psf=psf_obs,
        )


    def _get_offsets(self, offset_pixels):
        if offset_pixels is None:
            xoffset, yoffset = 0.0, 0.0
        else:
            xoffset = offset_pixels['col_offset']
            yoffset = offset_pixels['row_offset']

        return galsim.PositionD(xoffset, yoffset)

    def _process_obs(self):
        """
        add observations as interpolated images

        also keep track of psfs, variances, and noise realizations
        """
        self.images = []
        self.psfs = []
        self.vars = np.zeros(len(self.observations))
        self.noise_images = []

        self._set_coadd_obs()

        for i,obs in enumerate(self.observations):

            offset = self._get_offsets(obs.meta['offset_pixels'])
            psf_offset = self._get_offsets(obs.psf.meta['offset_pixels'])
            image_center = self.canonical_center + offset
            psf_image_center = self.psf_canonical_center + psf_offset

            # interplated image, shifted to center of the postage stamp
            jac = obs.jacobian

            wcs = galsim.TanWCS(
                affine=galsim.AffineTransform(
                    jac.dudcol,
                    jac.dudrow,
                    jac.dvdcol,
                    jac.dvdrow,
                    origin=image_center,
                ),
                world_origin=self.sky_center,
            )
            pjac = obs.psf.jacobian
            psf_wcs = galsim.TanWCS(
                affine=galsim.AffineTransform(
                    pjac.dudcol,
                    pjac.dudrow,
                    pjac.dvdcol,
                    pjac.dvdrow,
                    origin=psf_image_center,
                ),
                world_origin=self.sky_center,
            )

            image = galsim.InterpolatedImage(
                galsim.Image(obs.image,wcs=wcs),
                offset=offset,
                x_interpolant=self.interp,
            )

            # always normalizing psf
            psf_image = obs.psf.image.copy()
            psf_image /= psf_image.sum()

            psf = galsim.InterpolatedImage(
                galsim.Image(psf_image,wcs=psf_wcs),
                offset=psf_offset,
                x_interpolant=self.interp,
            )

            self.images.append(image)

            self.psfs.append(psf)

            # assume variance is constant
            var = 1./obs.weight.max()
            self.vars[i] = var

            # use input noise image
            noise_image = galsim.InterpolatedImage(
                galsim.Image(obs.noise,wcs=wcs),
                offset=offset,
                x_interpolant=self.interp,
            )

            self.noise_images.append(noise_image)

        self.weights = 1./self.vars
        self.weights /= self.weights.sum()
