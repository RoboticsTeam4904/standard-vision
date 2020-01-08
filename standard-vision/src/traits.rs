use crate::types::*;
use ndarray::{ArrayView3, ArrayViewMut3};
use std::io;

/// Camera which captures images of type `I` 
pub trait Camera<T, I: ImageData<T>> {
    /// Gets the config of the camera.
    fn config(&self) -> &CameraConfig;
    /// General setup for camera, etc.
    fn calibrate(&self) -> io::Result<()> {
        Ok(())
    }
    /// Grabs next image from the camera.
    fn grab_frame(&mut self) -> io::Result<Image<T, I>>;
}

/// Generalized format for image data.
/// `T` is the raw format of the data.
pub trait ImageData<T> {
    /// Returns a view into the memory which contains the pixels of `ImageData`'s implementor in ndarray form.
    fn as_pixels(&self) -> ArrayView3<u8>;
    /// Returns a mutable view into the memory which contains the pixels of `ImageData`'s implementor in ndarray form.
    fn as_pixels_mut(&mut self) -> ArrayViewMut3<u8>;
    /// Returns raw data of image.
    fn as_raw(&self) -> &T;
    /// Returns mutable raw data of image.
    fn as_raw_mut(&mut self) -> &mut T;
}

/// Exctracts contours from an Image
pub trait ContourExtractor<T, I: ImageData<T>> {
    fn extract_from(&self, image: &Image<T, I>) -> Vec<Contour>;
}

/// Calculates some generlized data `T` from contours and their associated camera
pub trait ContourAnalyzer<T> {
    fn analyze(&self, config: &CameraConfig, contours: &Vec<Contour>) -> Vec<T>;
}
