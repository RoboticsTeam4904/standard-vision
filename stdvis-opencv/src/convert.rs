use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    borrow::{Borrow, BorrowMut},
};

use ndarray::{prelude::*, RawData};
use opencv::prelude::*;
use stdvis_core::{traits::ImageData, types::Image};

pub trait AsArrayView {
    fn as_array_view<Data: DataType>(&self) -> ArrayViewD<Data>;
    fn as_array_view_mut<Data: DataType>(&mut self) -> ArrayViewMutD<Data>;
}

fn mat_dims(mat: &Mat) -> IxDyn {
    let size = mat.mat_size();
    let ndims = size.dims().unwrap() as usize;
    let mut dim_sizes = (0..ndims).map(|d| size[d] as usize).collect::<Vec<_>>();

    let channels = mat.channels().unwrap();
    dim_sizes.push(channels as usize);

    IxDyn(&dim_sizes)
}

impl AsArrayView for Mat {
    fn as_array_view<Data: DataType>(&self) -> ArrayViewD<Data> {
        unsafe {
            ArrayView::from_shape_ptr(
                mat_dims(self),
                self.ptr(0).unwrap() as *const _ as *const Data,
            )
        }
    }

    fn as_array_view_mut<Data: DataType>(&mut self) -> ArrayViewMutD<Data> {
        unsafe {
            ArrayViewMut::from_shape_ptr(
                mat_dims(self),
                self.ptr_mut(0).unwrap() as *mut _ as *mut Data,
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

impl<'mat> Borrow<Mat> for MatView<'mat> {
    fn borrow(&self) -> &Mat {
        &self.mat
    }
}

impl<'mat> BorrowMut<Mat> for MatView<'mat> {
    fn borrow_mut(&mut self) -> &mut Mat {
        &mut self.mat
    }
}

pub trait AsMatView {
    fn as_mat_view(&self) -> MatView;
}

impl<E, S, D> AsMatView for ArrayBase<S, D>
where
    E: DataType,
    S: RawData<Elem = E>,
    D: Dimension,
{
    fn as_mat_view(&self) -> MatView {
        let mut sizes = self
            .shape()
            .iter()
            .map(|size| size.to_owned() as i32)
            .collect::<Vec<_>>();

        let mut typ = <S::Elem as DataType>::typ();
        let channels = sizes.pop().unwrap();
        typ = opencv::core::CV_MAKETYPE(typ, channels);

        let strides = self
            .strides()
            .iter()
            .take(sizes.len() - 1)
            .map(|stride| stride.to_owned() as usize)
            .collect::<Vec<_>>();

        MatView::new(
            Mat::new_nd_with_data(
                &sizes,
                typ,
                unsafe { &mut *(self.as_ptr() as *mut std::ffi::c_void) },
                // TODO: New version of `opencv` uses only one stride...
                &strides[0],
            )
            .unwrap(),
        )
    }
}

impl<'src, I: ImageData> AsMatView for Image<'src, I> {
    fn as_mat_view(&self) -> MatView {
        MatView::new(self.as_pixels().as_mat_view().into())
    }
}

// TODO: Add tests
