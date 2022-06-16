from cProfile import label
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

ENTROPY = 0
VARIANCE = 1
GRADIENT = 2
AVG_EDGE_STRENGTH = 3
AVG_EDGE_STRENGTH_PATCH = 4
LAPLACIAN_VAR = 5
SOBEL_VAR = 6

FRAME_LEN = 0.05
PEAK_COOLDOWN = 60  # there can't be 2 peaks in 60 frames (3s)
SCALE_FACTOR = 0.25


class CameraFocus:
    def __init__(self):
        self.path = "./camera_focus_test_images"

    def blur_and_save_image(self, imgname):
        img = cv2.imread("{}/{}".format(self.path, imgname))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_name = imgname[:-4]
        extension = imgname[-4:]
        for ksize in [15, 55, 95, 135, 355, 565]:
            blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
            cv2.imwrite(
                "{}/{}_gaus_{:03}{}".format(self.path, img_name, ksize, extension), blur
            )

    def focus_entropy(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marg = np.histogramdd(np.ravel(img), bins=10)[0] / img.size
        marg = list(filter(lambda p: p > 0, np.ravel(marg)))
        entropy = -np.sum(np.multiply(marg, np.log2(marg)))
        return entropy

    def focus_variance(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.var(img)

    def focus_image_gradient(self, image):
        "Get magnitude of gradient for given image"
        grad, min_grad, max_grad = self.sobel(image)
        # mag = cv2.magnitude(dx, dy) / 1.414
        # mag = np.clip(mag, 0, 255)
        marg = np.histogram(np.ravel(grad), bins=np.arange(257))[0]
        # marg[:50] = np.zeros(50)
        val = grad.sum() / float(image.shape[0] * image.shape[1]) / 255.0
        plt.rcParams["figure.figsize"] = [16, 8]
        plt.subplot(1, 2, 1)
        plt.plot(
            np.arange(256),
            marg,
            label="std: {}\nmin: {}\nmax: {}\nsharpness: {}".format(
                np.std(grad), min_grad, max_grad, val
            ),
        )

        plt.legend(loc="upper left")
        plt.subplot(1, 2, 2)
        plt.imshow(grad)
        plt.show()
        return val

    def sobel(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(img, ddepth, 1, 0, ksize=3)
        dy = cv2.Sobel(img, ddepth, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(dx)
        abs_grad_y = cv2.convertScaleAbs(dy)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # self.NMS(grad, k=7)
        # check range of grad, then select filter criteria
        max_grad = np.max(grad)
        min_grad = np.min(grad)
        return grad, min_grad, max_grad

    def focus_average_edge_strength(self, img):
        """
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.7.9921&rep=rep1&type=pdf
        """
        # do edge detection
        # non-max suppression
        # take average of edge
        grad, min_grad, max_grad = self.sobel(img)
        filter_th = int((max_grad - min_grad) * 0.6) + min_grad
        grad = np.where(grad >= filter_th, grad, 0.0)
        num_valid_pix = np.sum(np.where(grad != 0, 1.0, 0.0), dtype=float)
        return np.sum(grad, dtype=float) / num_valid_pix if num_valid_pix > 0 else 0

    def focus_laplacian_var(self, img):
        image = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def focus_sobel_var(self, img):
        image = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        grad, min_grad, max_grad = self.sobel(image)
        return grad.var()

    def focus_average_edge_strength_patch(self, img):
        # scale image
        image = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        # do sobel
        return self.focus_average_edge_strength(image)
        # find patches containing edges

        # find AES of each patch
        # pass

    def test_focus_video(self, video_name, type=AVG_EDGE_STRENGTH):
        # get each frame, display it, compute its sharpness
        print("Computing sharpness from video..")
        cap = cv2.VideoCapture(video_name)
        if cap.isOpened() == False:
            raise Exception("Error opening video file. Exiting..")
        peak_times = []
        values = []
        total_time = 0.0
        frame_counter = 0
        peak_counter = 0
        peak_triggered = False
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow("Frame", frame)
                start_time = time.time()
                val = self.test_focus_image(frame, type)
                values.append(val)
                end_time = time.time()
                total_time += end_time - start_time
                key = cv2.waitKey(50)
                if peak_counter == PEAK_COOLDOWN:
                    peak_counter = 0
                    peak_triggered = False
                if key == ord("m"):
                    if peak_counter == 0:
                        peak_times.append(frame_counter)
                        print("marked at frame {}".format(frame_counter))
                        peak_triggered = True
                if peak_triggered == True:
                    peak_counter += 1
                frame_counter += 1
            elif ret == False:
                print("Reached end of video or got bad frame.")
                break
        cap.release()
        cv2.destroyAllWindows()
        print(
            "Done getting images from video. Average computation time: {:.4f} ms".format(
                total_time / float(frame_counter) * 1000.0
            )
        )
        values = np.array(values)
        # plot time vs sharpness amd save
        plt.plot(np.arange(frame_counter) * FRAME_LEN, values, label="sharpness")
        # plot manually-marked peak time and value
        marked_peak_arr = np.array(peak_times)
        marked_peak_val_arr = values[peak_times]
        plt.vlines(
            x=marked_peak_arr * FRAME_LEN,
            ymin=0,
            ymax=marked_peak_val_arr,
            colors="green",
            ls=":",
            lw=2,
            label="marked peaks",
        )
        plt.legend(loc="upper left")
        plt.show()

    def test_focus_image_file(self, img_name, type=AVG_EDGE_STRENGTH):
        fnames = os.listdir(self.path)
        fnames.sort()
        img_names = []
        sharpness_values = []
        for fname in fnames:
            if fname.find(img_name) != -1:
                abs_fname = "{}/{}".format(self.path, fname)
                try:
                    image = cv2.imread(abs_fname)
                except:
                    continue
                val = 0
                start_time = time.time()
                val = self.test_focus_image(image, type)
                end_time = time.time()
                print(
                    "Image: {}\ttime: {:.4f} ms\tFocus level: {}".format(
                        fname, (end_time - start_time) * 1000, val
                    )
                )
                img_names.append(abs_fname.split("/")[-1])
                sharpness_values.append(val)
        return img_names, sharpness_values

    def test_focus_image(self, image, type=AVG_EDGE_STRENGTH):
        if type == ENTROPY:
            val = self.focus_entropy(image)
        elif type == VARIANCE:
            val = self.focus_variance(image)
        elif type == GRADIENT:
            val = self.focus_image_gradient(image)
        elif type == AVG_EDGE_STRENGTH:
            val = self.focus_average_edge_strength(image)
        elif type == AVG_EDGE_STRENGTH_PATCH:
            val = self.focus_average_edge_strength_patch(image)
        elif type == LAPLACIAN_VAR:
            val = self.focus_laplacian_var(image)
        elif type == SOBEL_VAR:
            val = self.focus_sobel_var(image)
        return val

    def create_blank(self, _width, _height, _rgb_color=(0, 0, 0)):
        "Create new image(numpy array) filled with certain color in RGB."
        image = np.zeros((_height, _width, 3), np.uint8)
        color = tuple(reversed(_rgb_color))
        image[:] = color
        return image

    def create_blank_image_file(self, imgname, width, height, rgb_color=(0, 0, 0)):
        img = self.create_blank(width, height, _rgb_color=rgb_color)
        cv2.imwrite("{}/{}".format(self.path, imgname), img)

    def generate_ground_truth_data(self, out_file):
        # read all images, compute sharpness values, write to a .txt file
        names, sharpness = self.test_focus("png", type=AVG_EDGE_STRENGTH)
        maxlen = len(max(names, key=len))
        st = ""
        for name, sp in zip(names, sharpness):
            st = st + "{}\t{:.4f}\n".format(name.ljust(maxlen, " "), sp)
        with open(out_file, "w") as f:
            f.write(st)


if __name__ == "__main__":
    camera_focus = CameraFocus()
    # camera_focus.create_blank_image_file("black.png", 2880, 1860)
    camera_focus.test_focus_video("focus1.mp4", type=SOBEL_VAR)
    # camera_focus.generate_ground_truth_data("gt.txt")
    # camera_focus.test_focus("small_ok_1")
