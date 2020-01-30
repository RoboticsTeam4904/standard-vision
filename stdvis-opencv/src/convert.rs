use std::{marker::PhantomData, ops::{Deref, DerefMut}};

use ndarray::{prelude::*, ArrayView3, ArrayViewMut3, RawData};
use opencv::prelude::*;
use stdvis_core::{traits::ImageData, types::Image};

// TODO: Make conversions generalizable to n-dimensions, rather than only 3D.

pub trait AsArrayView {
    fn as_array_view(&self) -> ArrayView3<u8>;
    fn as_array_view_mut(&mut self) -> ArrayViewMut3<u8>;
}

impl AsArrayView for Mat {
    fn as_array_view(&self) -> ArrayView3<u8> {
        unsafe {
            ArrayView::from_shape_ptr(
                (
                    self.rows().unwrap() as usize,
                    self.cols().unwrap() as usize,
                    self.channels().unwrap() as usize,
                ),
                self.ptr(0).unwrap(),
            )
        }
    }

    fn as_array_view_mut(&mut self) -> ArrayViewMut3<u8> {
        unsafe {
            ArrayViewMut::from_shape_ptr(
                (
                    self.rows().unwrap() as usize,
                    self.cols().unwrap() as usize,
                    self.channels().unwrap() as usize,
                ),
                self.ptr_mut(0).unwrap(),
            )
        }
    }
}

pub struct MatView<'mat> {
    mat: Mat,
    phantom: PhantomData<&'mat Mat>,
}

impl<'mat> MatView<'mat> {
    fn new(mat: Mat) -> Self {
        Self {
            mat,
            phantom: PhantomData,
        }
    }
}

impl<'mat> Deref for MatView<'mat> {
    type Target = Mat;

    fn deref(&self) -> &Mat {
        &self.mat
    }
}

impl<'mat> DerefMut for MatView<'mat> {
    fn deref_mut(&mut self) -> &mut Mat {
        &mut self.mat
    }
}

pub trait AsMatView {
    fn as_mat_view(&self) -> Mat;
}

impl<S, D> AsMatView for ArrayBase<S, D>
where
    S: RawData,
    D: Dimension,
{
    fn as_mat_view(&self) -> Mat {
        Mat::new_rows_cols_with_data(
            self.len_of(Axis(0)) as i32,
            self.len_of(Axis(1)) as i32,
            opencv::core::CV_8UC3,
            unsafe { &mut *(self.as_ptr() as *mut std::ffi::c_void) },
            self.stride_of(Axis(0)) as usize,
        )
        .unwrap()
    }
}

impl<I: ImageData> AsMatView for Image<I> {
    fn as_mat_view(&self) -> Mat {
        self.as_pixels().as_mat_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random() {
        let mut ndarray = array![[0, 1], [1, 0]];

        println!("{:?}", ndarray);
    }
}
