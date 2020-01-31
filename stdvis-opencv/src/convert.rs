use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use ndarray::{prelude::*, ArrayViewD, ArrayViewMutD, Dimension, RawData};
use opencv::{prelude::*, types::VectorOfint};
use stdvis_core::{traits::ImageData, types::Image};

pub trait AsArrayView {
    fn mat_dims(&self) -> IxDyn;
    fn as_array_view(&self) -> ArrayView<u8, IxDyn>;
    fn as_array_view_mut(&mut self) -> ArrayViewMut<u8, IxDyn>;
}

impl AsArrayView for Mat {
    fn mat_dims(&self) -> IxDyn {
        let size = self.mat_size().unwrap();
        let ndims = size.dims().unwrap() as usize;
        let mut dim_sizes = (0..ndims).map(|d| size[d] as usize).collect::<Vec<_>>();
        let channels = self.channels().unwrap();
        
        if channels > 0 { dim_sizes.push(channels as usize); }

        IxDyn(&dim_sizes)
    }

    fn as_array_view(&self) -> ArrayViewD<u8> {
        unsafe { ArrayView::from_shape_ptr(self.mat_dims(), self.ptr(0).unwrap()) }
    }

    fn as_array_view_mut(&mut self) -> ArrayViewMutD<u8> {
        unsafe { ArrayViewMut::from_shape_ptr(self.mat_dims(), self.ptr_mut(0).unwrap()) }
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

impl<'mat> Into<Mat> for MatView<'mat> {
    fn into(self) -> Mat {
        self.mat
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
    fn as_mat_view(&self, channels: u32) -> MatView;
}

impl<S, D> AsMatView for ArrayBase<S, D>
where
    S: RawData,
    D: Dimension,
{
    fn as_mat_view(&self, channels: u32) -> MatView {
        let mut sizes = VectorOfint::from_iter(self.shape().iter().map(|size| size.to_owned() as i32));

        if channels > 0 {
            sizes.remove(sizes.len() - 1).unwrap();
        }

        let strides = self
            .strides()
            .iter()
            .map(|stride| stride.to_owned() as usize)
            .collect::<Vec<_>>();

        MatView::new(
            Mat::new_nd_with_data(
                &sizes,
                opencv::core::CV_MAKETYPE(
                    opencv::core::CV_8U,
                    channels as i32,
                ),
                unsafe { &mut *(self.as_ptr() as *mut std::ffi::c_void) },
                &strides,
            )
            .unwrap(),
        )
    }
}

impl<I: ImageData> AsMatView for Image<I> {
    fn as_mat_view(&self, channels: u32) -> MatView {
        MatView::new(self.as_pixels().as_mat_view(channels).into())
    }
}

// TODO: Add tests
