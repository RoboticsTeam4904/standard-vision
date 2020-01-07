#[cfg(test)]
extern crate png;

use std::{io, marker::PhantomData};

use ndarray::{ArrayView3, ArrayViewMut, ArrayViewMut3};
use opencv::prelude::*;
use opencv::videoio::*;
use standard_vision::traits::{Camera, ImageData};
use standard_vision::types::{CameraConfig, Image, Pose};

struct OpenCVImage<'a> {
    mat: Mat,
    pixels: ArrayViewMut3<'a, u8>,
}

impl<'a> ImageData<Mat> for OpenCVImage<'a> {
    fn as_pixels(&self) -> ArrayView3<u8> {
        self.pixels.view()
    }

    fn as_pixels_mut(&mut self) -> ArrayViewMut3<u8> {
        self.pixels.view_mut()
    }

    fn as_raw(&self) -> &Mat {
        &self.mat
    }

    fn as_raw_mut(&mut self) -> &mut Mat {
        &mut self.mat
    }
}

pub struct OpenCVCamera {
    config: CameraConfig,
    video_capture: VideoCapture,
}

impl OpenCVCamera {
    pub fn new_from_index(index: i32, pose: Pose, fov: f64, focal_length: f64) -> io::Result<Self> {
        let mut video_capture = VideoCapture::default().unwrap();

        if !video_capture.open_with_backend(index, CAP_ANY).unwrap() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Failed to open camera on port {}", index),
            ));
        }

        Self::new_from_video_capture(video_capture, index, pose, fov, focal_length)
    }

    pub(crate) fn new_from_video_capture(
        video_capture: VideoCapture,
        index: i32,
        pose: Pose,
        fov: f64,
        focal_length: f64,
    ) -> io::Result<Self> {
        let resolution = (
            video_capture.get(CAP_PROP_FRAME_WIDTH).unwrap() as u32,
            video_capture.get(CAP_PROP_FRAME_HEIGHT).unwrap() as u32,
        );

        let config = CameraConfig {
            id: index as u8,
            resolution,
            pose,
            fov,
            focal_length,
        };

        Ok(Self {
            config,
            video_capture,
        })
    }

    pub(crate) unsafe fn extract_pixels_from_mat<'a>(mat: Mat) -> OpenCVImage<'a> {
        let mut mat = mat;
        let pixels = ArrayViewMut::from_shape_ptr(
            (
                mat.rows().unwrap() as usize,
                mat.cols().unwrap() as usize,
                3,
            ),
            mat.ptr_mut(0).unwrap(),
        );
        OpenCVImage { mat, pixels }
    }
}

impl<'a> Camera<Mat, OpenCVImage<'a>> for OpenCVCamera {
    fn get_config(&self) -> &CameraConfig {
        &self.config
    }

    fn grab_frame(&mut self) -> io::Result<Image<Mat, OpenCVImage<'a>>> {
        let mut mat = Mat::default().unwrap();
        if !self.video_capture.read(&mut mat).unwrap() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Failed to read frame from camera at port {}",
                    self.config.id
                ),
            ));
        }

        let pixels = unsafe { Self::extract_pixels_from_mat(mat) };

        Ok(Image::new(
            std::time::SystemTime::now(),
            self.get_config(),
            pixels,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_pixel_extraction() {
        use opencv::imgcodecs;
        use std::fs::File;

        const PATH: &str = "tests/images/rand.png";

        let decoder = png::Decoder::new(File::open(PATH).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut img_buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut img_buf).unwrap();

        let expected_pixels = img_buf
            .chunks_exact(3)
            .map(|items| [items[2], items[1], items[0]])
            .collect::<Vec<_>>();

        let mat = imgcodecs::imread(PATH, imgcodecs::IMREAD_COLOR).unwrap();

        let actual_pixels = OpenCVCamera::extract_pixels_from_mat(mat);
        assert_eq!(expected_pixels, actual_pixels);
    }
}
