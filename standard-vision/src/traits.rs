use crate::types::*;
use ndarray::{ArrayView3, ArrayViewMut3};
use std::io;

pub trait Camera<T, I: ImageData<T>> {
    fn get_config(&self) -> &CameraConfig;
    fn calibrate(&self) -> io::Result<()> {
        Ok(())
    }
    fn grab_frame(&mut self) -> io::Result<Image<T, I>>;
}

pub trait ImageData<T> {
    fn as_pixels(&self) -> ArrayView3<u8>;
    fn as_pixels_mut(&mut self) -> ArrayViewMut3<u8>;
    fn as_raw(&self) -> &T;
    fn as_raw_mut(&mut self) -> &mut T;
}

pub trait ContourExtractor<T, I: ImageData<T>> {
    fn extract_from(&self, image: &Image<T, I>) -> Vec<Contour>;
}

pub trait ContourAnalyzer {
    fn analyze(&self, config: &CameraConfig, contours: &Vec<Contour>) -> Vec<Target>;
}
