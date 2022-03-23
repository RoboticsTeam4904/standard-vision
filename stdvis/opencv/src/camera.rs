use std::io;

use ndarray::{ArrayViewD, ArrayViewMutD};
use opencv::{core::Vector, prelude::*, videoio::*};
use stdvis_core::{
    traits::{Camera, ImageData},
    types::{CameraConfig, Image},
};
use v4l::{Control, Device};

#[cfg(feature = "cuda")]
use opencv::{
    core::{GpuMat, Ptr, Stream},
    cudacodec::{create_video_reader, VideoReader},
};

use crate::convert::AsArrayView;

pub struct MatImageData {
    mat: Mat,
}

impl MatImageData {
    pub fn new(mat: Mat) -> Self {
        Self { mat }
    }
}

impl ImageData for MatImageData {
    type Inner = Mat;

    fn as_pixels(&self) -> ArrayViewD<u8> {
        self.mat.as_array_view()
    }

    fn as_pixels_mut(&mut self) -> ArrayViewMutD<u8> {
        self.mat.as_array_view_mut()
    }

    fn as_raw(&self) -> &Self::Inner {
        &self.mat
    }

    fn as_raw_mut(&mut self) -> &mut Self::Inner {
        &mut self.mat
    }
}

pub struct OcvCamera {
    config: CameraConfig,

    device: Device,

    #[cfg(not(any(feature = "cuda")))]
    video_source: VideoCapture,

    #[cfg(feature = "cuda")]
    video_source: Ptr<dyn VideoReader>,
}

impl OcvCamera {
    pub fn new(config: CameraConfig) -> io::Result<Self> {
        // TODO: better error handling

        let id = config.id;
        let device_path = format!("/dev/video{id}");

        let device = Device::with_path(&device_path)?;

        let params = &[
            CAP_PROP_FRAME_WIDTH,
            config.resolution.0 as i32,
            CAP_PROP_FRAME_HEIGHT,
            config.resolution.1 as i32,
        ];

        #[cfg(not(any(feature = "cuda")))]
        let video_source = VideoCapture::new_with_params(
            id as i32,
            {
                if cfg!(target_os = "linux") {
                    CAP_V4L
                } else {
                    CAP_ANY
                }
            },
            &Vector::from_slice(params),
        )
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

        #[cfg(feature = "cuda")]
        let video_source = create_video_reader(&device_path, &Vector::from_slice(params), false)
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;

        Ok(Self {
            device,
            config,
            video_source,
        })
    }

    pub fn exposure(&self) -> io::Result<i32> {
        use v4l::v4l_sys::V4L2_CID_EXPOSURE;

        match self.device.control(V4L2_CID_EXPOSURE)? {
            Control::Value(exposure) => Ok(exposure),
            _ => panic!("unexpected control type"),
        }
    }

    pub fn set_exposure(&mut self, exposure: i32) -> io::Result<()> {
        use v4l::v4l_sys::{
            v4l2_exposure_auto_type_V4L2_EXPOSURE_MANUAL, V4L2_CID_EXPOSURE, V4L2_CID_EXPOSURE_AUTO,
        };

        self.device.set_control(
            V4L2_CID_EXPOSURE_AUTO,
            Control::Value(v4l2_exposure_auto_type_V4L2_EXPOSURE_MANUAL as i32),
        )?;
        self.device
            .set_control(V4L2_CID_EXPOSURE, Control::Value(exposure))?;

        Ok(())
    }
}

impl Camera for OcvCamera {
    type ImageStorage = MatImageData;

    fn config(&self) -> &CameraConfig {
        &self.config
    }

    fn grab_frame(&mut self) -> io::Result<Image<Self::ImageStorage>> {
        let mut mat = Mat::default();

        #[cfg(feature = "cuda")]
        let mut gpu_mat = GpuMat::default().expect("initializing GpuMat");

        #[cfg(not(any(feature = "cuda")))]
        let success = self
            .video_source
            .read(&mut mat)
            .expect("reading from VideoCapture");

        #[cfg(feature = "cuda")]
        let success = self
            .video_source
            .next_frame(&mut gpu_mat, &mut Stream::null().unwrap())
            .expect("reading from VideoReader");

        if !success {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Failed to read frame from camera at port {}",
                    self.config.id
                ),
            ));
        }

        #[cfg(feature = "cuda")]
        gpu_mat
            .download(&mut mat)
            .expect("downloading GpuMat to Mat");

        // TODO: timestamp should be taken between grab() and retrieve() calls

        Ok(Image::new(
            std::time::Instant::now(),
            self.config(),
            MatImageData::new(mat),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ndarray::s;
    use opencv::{core::Vec3b, imgcodecs};

    use super::*;

    #[test]
    fn test_mat_pixel_extraction() {
        use crate::convert::AsMatView;

        const PATH: &str = "tests/images/rand.png";

        let decoder = png::Decoder::new(File::open(PATH).unwrap());
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut img_buf = vec![0; info.buffer_size()];
        reader.next_frame(&mut img_buf).unwrap();

        let expected_pixels_buf = img_buf
            .chunks_exact(3)
            .map(|items| [items[2], items[1], items[0]])
            .collect::<Vec<_>>();

        let cv_image = MatImageData::new(imgcodecs::imread(PATH, imgcodecs::IMREAD_COLOR).unwrap());

        let config = CameraConfig::default();
        let image = Image::new(std::time::Instant::now(), &config, cv_image);

        let cv_pixels = image.as_pixels();
        let cv_raw = &image.as_mat_view();

        assert_eq!(
            cv_pixels.shape(),
            [
                cv_raw.rows() as usize,
                cv_raw.cols() as usize,
                cv_raw.channels() as usize,
            ]
        );

        let img_dims = cv_pixels.shape();
        for row in 0..img_dims[0] {
            for col in 0..img_dims[1] {
                let expected_pixel = expected_pixels_buf[row * img_dims[1] + col];

                assert_eq!(
                    cv_pixels.slice(s![row, col, ..]).as_slice().unwrap(),
                    expected_pixel
                );

                assert_eq!(
                    **cv_raw.at_2d::<Vec3b>(row as i32, col as i32).unwrap(),
                    expected_pixel
                );
            }
        }
    }
}
