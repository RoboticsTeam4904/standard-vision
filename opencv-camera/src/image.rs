use ndarray::prelude::*;
use opencv::prelude::*;

use standard_vision::{traits::ImageData, types::Image};

pub trait AsMat {
    fn as_mat(&self) -> Mat;
    fn as_mat_mut(&mut self) -> Mat;
}

impl<'a, I: ImageData> AsMat for Image<'a, I> {
    fn as_mat(&self) -> Mat {
        let pixels = self.as_pixels();

        Mat::new_rows_cols_with_data(
            pixels.len_of(Axis(0)) as i32,
            pixels.len_of(Axis(1)) as i32,
            opencv::core::CV_8UC3,
            unsafe { &mut *(pixels.first().unwrap() as *const u8 as *mut std::ffi::c_void) },
            3,
        ).unwrap()
    }

    fn as_mat_mut(&mut self) -> Mat {
        let mut pixels = self.as_pixels_mut();

        Mat::new_rows_cols_with_data(
            pixels.len_of(Axis(0)) as i32,
            pixels.len_of(Axis(1)) as i32,
            opencv::core::CV_8UC3,
            unsafe { &mut *(pixels.first_mut().unwrap() as *mut u8 as *mut std::ffi::c_void) },
            3,
        ).unwrap()
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
