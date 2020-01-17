#[cfg(test)]
use png;

use std::io;

use ndarray::{ArrayView3, ArrayViewMut, ArrayViewMut3};
use opencv::{prelude::*, videoio::*};
use standard_vision::{
    traits::{Camera, ImageData},
    types::{CameraConfig, Image, Pose},
};

pub mod image;

pub struct OpenCVImage<'a> {
    mat: Mat,
    pixels: ArrayViewMut3<'a, u8>,
}

impl<'a> ImageData for OpenCVImage<'a> {
    type Inner = Mat;

    fn as_pixels(&self) -> ArrayView3<u8> {
        self.pixels.view()
    }

    fn as_pixels_mut(&mut self) -> ArrayViewMut3<u8> {
        self.pixels.view_mut()
    }

    fn as_raw(&self) -> &Self::Inner {
        &self.mat
    }

    fn as_raw_mut(&mut self) -> &mut Self::Inner {
        &mut self.mat
    }
}

pub struct OpenCVCamera {
    config: CameraConfig,
    video_capture: VideoCapture,
}

impl OpenCVCamera {
    pub fn new_from_index(
        index: i32,
        pose: Pose,
        fov: f64,
        focal_length: f64,
        sensor_height: f64,
    ) -> io::Result<Self> {
        let mut video_capture = VideoCapture::default().unwrap();

        if !video_capture.open_with_backend(index, CAP_ANY).unwrap() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Failed to open camera on port {}", index),
            ));
        }

        Self::new_from_video_capture(video_capture, index, pose, fov, focal_length, sensor_height)
    }

    pub(crate) fn new_from_video_capture(
        video_capture: VideoCapture,
        index: i32,
        pose: Pose,
        fov: f64,
        focal_length: f64,
        sensor_height: f64,
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
            sensor_height,
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

impl<'a> Camera<OpenCVImage<'a>> for OpenCVCamera {
    fn config(&self) -> &CameraConfig {
        &self.config
    }

    fn grab_frame(&mut self) -> io::Result<Image<OpenCVImage<'a>>> {
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
            self.config(),
            pixels,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_pixel_extraction() {
        use ndarray::s;
        use opencv::{core::Vec3, imgcodecs};
        use std::fs::File;

        const PATH: &str = "tests/images/rand.png";

        let decoder = png::Decoder::new(File::open(PATH).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut img_buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut img_buf).unwrap();

        let expected_pixels_buf = img_buf
            .chunks_exact(3)
            .map(|items| [items[2], items[1], items[0]])
            .collect::<Vec<_>>();

        let cv_image = unsafe {
            OpenCVCamera::extract_pixels_from_mat(
                imgcodecs::imread(PATH, imgcodecs::IMREAD_COLOR).unwrap(),
            )
        };
        let cv_pixels = cv_image.as_pixels();
        let cv_raw = cv_image.as_raw();
        assert_eq!(
            cv_pixels.shape(),
            [
                cv_raw.rows().unwrap() as usize,
                cv_raw.cols().unwrap() as usize,
                3
            ]
        );
        let img_dims = cv_pixels.shape();
        for i in 0..img_dims[0] {
            for j in 0..img_dims[1] {
                let expected_pixel = expected_pixels_buf[i * img_dims[1] + j];
                assert_eq!(
                    cv_pixels.slice(s![i, j, ..]).as_slice().unwrap(),
                    expected_pixel
                );
                assert_eq!(
                    **cv_raw.at_2d::<Vec3<u8>>(i as i32, j as i32).unwrap(),
                    expected_pixel
                );
            }
        }
    }
}
