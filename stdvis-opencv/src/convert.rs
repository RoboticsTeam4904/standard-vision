use ndarray::{ArrayView3, ArrayViewMut3, prelude::*};
use opencv::prelude::*;
use stdvis_core::{traits::ImageData, types::Image};

pub fn mat_array_view<'a>(mat: &'a Mat) -> ArrayView3<'a, u8> {
    unsafe {
        ArrayView::from_shape_ptr(
            (
                mat.rows().unwrap() as usize,
                mat.cols().unwrap() as usize,
                mat.channels().unwrap() as usize,
            ),
            mat.ptr(0).unwrap(),
        )
    }
}

pub fn mat_array_view_mut<'a>(mat: &'a mut Mat) -> ArrayViewMut3<'a, u8> {
    unsafe {
        ArrayViewMut::from_shape_ptr(
            (
                mat.rows().unwrap() as usize,
                mat.cols().unwrap() as usize,
                mat.channels().unwrap() as usize,
            ),
            mat.ptr_mut(0).unwrap(),
        )
    }
}

pub trait AsMat {
    fn as_mat(&self) -> Mat;
    fn as_mat_mut(&mut self) -> Mat;
}

impl<I: ImageData> AsMat for Image<I> {
    fn as_mat(&self) -> Mat {
        let pixels = self.as_pixels();

        Mat::new_rows_cols_with_data(
            pixels.len_of(Axis(0)) as i32,
            pixels.len_of(Axis(1)) as i32,
            opencv::core::CV_8UC3,
            unsafe { &mut *(pixels.as_ptr() as *mut std::ffi::c_void) },
            pixels.stride_of(Axis(0)) as usize,
        )
        .unwrap()
    }

    fn as_mat_mut(&mut self) -> Mat {
        let mut pixels = self.as_pixels_mut();

        Mat::new_rows_cols_with_data(
            pixels.len_of(Axis(0)) as i32,
            pixels.len_of(Axis(1)) as i32,
            opencv::core::CV_8UC3,
            unsafe { &mut *(pixels.as_mut_ptr() as *mut std::ffi::c_void) },
            pixels.stride_of(Axis(0)) as usize,
        )
        .unwrap()
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
