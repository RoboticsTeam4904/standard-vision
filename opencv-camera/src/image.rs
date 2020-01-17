use opencv::prelude::*;
use standard_vision::{traits::ImageData, types::Image};

trait AsMat {
    fn as_mat(&self) -> Mat;
    fn as_mat_mut(&mut self) -> Mat;
}

impl<'a, T: ImageData> AsMat for Image<'a, T> {
    fn as_mat(&self) -> Mat {
        unsafe { Mat::from_raw_ptr(self.as_pixels().as_ptr() as *mut std::ffi::c_void) }
    }

    fn as_mat_mut(&mut self) -> Mat {
        unsafe { Mat::from_raw_ptr(self.as_pixels().as_ptr() as *mut std::ffi::c_void) }
    }
}

// Trait specialization now plz

// impl<'a> AsMat for Image<'a, OpenCVImage> {
//     fn as_mat(&self) -> &Mat {
//         self.as_raw()
//     }

//     fn as_mat_mut(&mut self) -> &Mat {
//         self.as_raw_mut()
//     }
// }
