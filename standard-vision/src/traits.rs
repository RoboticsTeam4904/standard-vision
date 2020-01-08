use crate::types::*;
use ndarray::{ArrayView3, ArrayViewMut3};
use std::io;

/// Camera which captures images of type `I` 
pub trait Camera<I: ImageData> {
    /// Gets the config of the camera.
    fn config(&self) -> &CameraConfig;
    /// General setup for camera, etc.
    fn calibrate(&self) -> io::Result<()> {
        Ok(())
    }
    /// Grabs next image from the camera.
    fn grab_frame(&mut self) -> io::Result<Image<I>>;
}

/// Generalized format for image data.
/// `T` is the raw format of the data.
pub trait ImageData {
    type Inner;

    /// Returns a view into the memory which contains the pixels of `ImageData`'s implementor in ndarray form.
    fn as_pixels(&self) -> ArrayView3<u8>;
    /// Returns a mutable view into the memory which contains the pixels of `ImageData`'s implementor in ndarray form.
    fn as_pixels_mut(&mut self) -> ArrayViewMut3<u8>;
    /// Returns raw data of image.
    fn as_raw(&self) -> &Self::Inner;
    /// Returns mutable raw data of image.
    fn as_raw_mut(&mut self) -> &mut Self::Inner;
}

/// Exctracts contours from an Image
pub trait ContourExtractor<I: ImageData> {
    fn extract_from(&self, image: &Image<I>) -> Vec<Contour>;
}

/// Calculates some generlized data `T` from contours and their associated camera
pub trait ContourAnalyzer<T> {
    fn analyze(&self, config: &CameraConfig, contours: &Vec<Contour>) -> Vec<T>;
}
