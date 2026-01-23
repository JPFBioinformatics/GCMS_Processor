# region Imports

import sys
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import matplotlib.pyplot as plt

# location of pipeline root dir
root_dir = Path(__file__).resolve().parent.parent
# tell python to look here for modules
sys.path.insert(0, str(root_dir))

from src.config_loader import ConfigLoader

# endregion

# Class for storage and cleaning of intensity matrix extracted by mzml_processor
class IntensityMatrix:

    def __init__(self, intensity_matrix: np.ndarray, unique_mzs: list, spectra_name: str = None, spectra_metadata: dict = None, matrix_type: str = None, cfg: ConfigLoader = ConfigLoader(root_dir / "config.yaml")):
        self.intensity_matrix = intensity_matrix
        self.unique_mzs = unique_mzs
        self.spectra_metadata = spectra_metadata
        self.noise_factor = None
        self.abundance_threshold = None
        self.peak_list = None
        self.cfg = cfg
        self.spectra_name = spectra_name
        self.matrix_type = matrix_type

        # calculate and apply abundnace threshold transformation to intensity matrix
        self.calculate_threshold()
        self.apply_threshold()
        # calculate noise factor for this intensity matrix
        self.calculate_noise_factor()
        # identify peaks in this intensity matrix
        self.identify_peaks(self.intensity_matrix)
        
    # region Getter/Setters
    @property
    def intensity_matrix(self):
        return self._intensity_matrix

    @intensity_matrix.setter
    def intensity_matrix(self,value):
        if isinstance(value,np.ndarray):
            self._intensity_matrix = value
        else:
            raise ValueError("intensity matrix is not a numpy array")

    @property
    def unique_mzs(self):
        return self._unique_mzs

    @unique_mzs.setter
    def unique_mzs(self,value):
        if not len(value) == self.intensity_matrix.shape[0]:
            raise ValueError(f"unique m/z length {len(value)} does not match intensity array row count {self.intensity_matrix.shape[0]}")
        if not isinstance(value, list):
            raise ValueError('unique m/z is not a list')
        else:
            self._unique_mzs = value

    @property
    def spectra_metadata(self):
        return self._spectra_metadata

    @spectra_metadata.setter
    def spectra_metadata(self,value):
        if value is not None:
            if not len(value) == self.intensity_matrix.shape[1]:
                raise ValueError('Spectra metadata length does not match intensity array column count')
            if not isinstance(value, dict):
                raise ValueError('Spectra metadata is not a list')
        self._spectra_metadata = value
    #endregion

    # region Abundance Threshold

    # calculates At value to replace 0 values with
    def calculate_threshold(self):

        """
        Counts the number of zero to nonzero transtions for each m/z in 10 approximately equally sized time segments then takes the square root
        of these values and multiplies it by the minimum abundance measured in the entire intensity matrix, use this value to replace 0 values

        Parameters:
            intensity_matrix (np.ndarray): 2D numpy array where each row corrosponds to a m/z chromatogram intensity profile and each column
                                           corrosponds to a scan
        Returns:
            threshold_values (np.ndarray): 2D numpy array with 10 columns (1 per segment) and a row for each unique m/z in the input matrix
                                           each entry corrosponds to the calculated threshold value for that m/z in that segment
        """

        intensity_matrix = self.intensity_matrix

        # the minimum measured abundance in the intensity matrix
        min_value = np.min(intensity_matrix[intensity_matrix>0])

        # creates a 2d array to store the threshold transitions, min value in the input array/sqrt(fraction of transitions in segment in row)
        threshold_values = np.empty((len(self.unique_mzs),10))

        # split the array into 10 approximately equal time segments
        segments = np.array_split(intensity_matrix, 10, axis=1)

        # counter for the start index of each segment
        start_idx = 0

        # list to hold the start index of each segment
        segment_starts = []

        # for each segment count the number of times a 0 value is followed by a nonzero value and store in transitions array
        for seg_idx,segment in enumerate(segments):
            for row_idx, row in enumerate(segment):

                # create array with 1 in any position where a 0 to nonzero transition occurs
                transitions = ((row[:-1] == 0) & (row[1:] > 0))
                # number of 0 to nonzero transitions in row
                num_transitions = np.sum(transitions)
                # length of the segment
                segment_length = segment.shape[1]
                # fraction of all scans in segment that are involved in m/z transtion
                threshold_values[row_idx, seg_idx] = num_transitions / segment_length

            # adds the start index for this segment to segment_starts list
            segment_starts.append(start_idx)
            # increments start_idx so that it now holds the first index value of the next segment
            start_idx += segment_length

        # takes the square root of all transition fraction values
        threshold_values **= 0.5

        # multiplies these square rooted values by the min value in matrix
        threshold_values *= min_value

        # dictionary that holds the start index of each segment (list) and the 2D numpy array (10 col, len(unique_mzs) rows) with threshold values stored in each cell
        threshold_dict = {
            'start_idxs' : segment_starts,
            'values' : threshold_values
        }
 
        self.abundance_threshold = threshold_dict

    # takes any value in the array that is below At for that segment for that m/z value and 
    def apply_threshold(self):

        matrix = self.intensity_matrix

        # iterate over each row, segment
        for row_idx,row in enumerate(matrix):
            for seg_idx in range(10):
                
                # start index for this segment
                start = self.abundance_threshold['start_idxs'][seg_idx]

                # end index for this segment
                if seg_idx <9:
                    end = self.abundance_threshold['start_idxs'][seg_idx+1]
                else:
                    end = row.shape[0]

                # get threshold value for this row this segment
                threshold = self.abundance_threshold['values'][row_idx,seg_idx]

                # replace values in this range for this segment with threshold
                row[start:end] = np.where(row[start:end]<threshold,threshold,row[start:end])

        self.intensity_matrix = matrix 

    # endregion

    # region Noise Factor Calculation

    # calculates the noise factor (Nf) for the entire intensity_matrix
    def calculate_noise_factor(self):

        matrix = self.intensity_matrix

        num_segments = matrix.shape[1] // 13
        segments = []
        noise_factors = []

        #loop over the number of segments creating each segment in segments as we go
        for i in range(num_segments):
            start = i*13
            end = (i+1)*13
            segment = matrix[:, start:end]
            
            # filters out any rows that contain 0 values
            nonzero_rows = segment[~np.any(segment == 0, axis = 1)]
            
            # filter rows that cross less than 7 times
            crossing_filtered = []
            for row in nonzero_rows:
                if row.size == 0:
                    continue
                avg = np.mean(row)

                # skip rows with 0 variation
                if avg == 0:
                    continue

                crossings = self.count_crossings(row,avg)
                if crossings > 6:
                    crossing_filtered.append(row)
                
            segments.append(np.array(crossing_filtered))

        # iterate through each segment
        for segment in segments:
            for row in segment:
                current_nf = self.calculate_row_nf(row)
                if not np.isnan(current_nf):
                    noise_factors.append(current_nf)

        # fallback if no noise factors calculated:
        if len(noise_factors) == 0:
            self.noise_factor = 0
        else:
            self.noise_factor = np.median(noise_factors)
    
    # counts the number of times the values of an array "cross" a given average value
    def count_crossings(self,row,avg):
        crossings = 0
        for i in range(len(row)-1):
            if (row[i] < avg and row[i+1] > avg) or (row[i] > avg and row[i+1] < avg):
                crossings += 1
        return crossings

    # calculates and returns the median deviation for a given 1D array
    def calculate_row_nf(self, row):

        # calculate the mean of the row
        mean = np.mean(row)
        if mean == 0:
            return 0
        
        # calculate rest of row nf 
        sqrt_of_mean = mean ** 0.5

        # calculate deviation from the mean for all members of row
        deviations = np.abs(row-mean)

        # calculat noise factor
        nf = np.median(deviations)/sqrt_of_mean
        
        # return the median of the deviations / sqrt of the mean (Nf for that row)
        return nf

    # endregion

    # region Finding Maxima

    # finds the peaks (maxima and bounds) for each row of a given intensity matrix and the tic, last row is TIC
    def identify_peaks(self, matrix, prom=None):

        # array to hold the lists of peak values, each entry of peaks corrosponds to a single m/z row in same order as unique_mzs
        peaks = []

        for row_idx,row in enumerate(matrix):

            ion = self.unique_mzs[row_idx]
            row_peaks = self.find_maxima(row,ion,prom=prom)
            peaks.append(row_peaks)

        self.peak_list = peaks

        return peaks

    # finds local maxima and bounds of peaks for a given 1D array
    def find_maxima(self, array, ion, prom = None):

        # set prominance
        if prom == None:
            # handle nan/0 noise factors (SIM files have this a lot)
            if np.isnan(self.noise_factor) or self.noise_factor == 0:
                # set based on median and mad (3*MAD is a common noise threshold hueristic)
                prom = np.median(array) + 3 * np.median(np.abs(array-np.median(array)))
            else:
                prom = self.noise_factor*100
            
        # Excludes the first and last 12 points from the search to prevent bounding errors
        range = array[12:-12]

        # finds the local maxima of the given array, stores their index
        max_idxs, _ = find_peaks(range, prominence=prom)

        # Shifts indices found in the range for use in the original array
        max_idxs += 12

        # list to hold dictionary entries containing left_bound, right_bound and center for each maxima
        maxima = []

        # go through each maxima in list, find its deconvolution window and check if sinal is high enough to be included
        for peak_max in max_idxs:

            # find the left bound of the deconvolution window
            left_bound_scan = self.find_bound(array,peak_max,-1)
            # find the right bound of the deconvolution window
            right_bound_scan = self.find_bound(array,peak_max,1)
            
            # width filter, skip peak if peak has less than 3 scans on either side of the peak
            if right_bound_scan - peak_max < 3 or peak_max - left_bound_scan < 3:
                continue
            
            # calculate baseline
            baseline = self.tentative_baseline(left_bound_scan, right_bound_scan, array)

            # calcluate quadratic fit for peak
            fit = self.quadratic_fit(array,peak_max)

            # grab the precise location and height of the peak
            precise_max_location = fit['x_values'][1]
            precise_max_height = fit['y_values'][1] - (baseline['slope']*precise_max_location + baseline['y_int'])
            precise_max_abundance = fit['y_values'][1]
            
            # finds the bin (0.1 of a scan) that the precise max is located within by truncating at 1 decimal point
            max_bin = int(precise_max_location*10) / 10

            # calculate convolution value for this peak
            conv = self.convolution_value(array,peak_max)
            
            max_info = {
                'left_bound' : left_bound_scan,
                'right_bound' : right_bound_scan,
                'center' : peak_max,
                'precise_max_location' : precise_max_location,
                'precise_max_height' : precise_max_height,
                'max_abundance' : precise_max_abundance,
                'bin' : max_bin,
                'conv_value' : conv,
                'ion' : ion,
                'tentative_baseline': baseline
            }

            # add width flag
            width = right_bound_scan - left_bound_scan
            max_info["width"] = width
            if width < 5:
                max_info["width_flag"] = "small"
            elif width < 10:
                max_info["width_flag"] = "ideal"
            elif width < 25:
                max_info["width_flag"] = "normal"
            else:
                max_info["width_flag"] = "overloaded"

            # add baseline/flat top flag
            if baseline == None:
                max_info["flat_top"] = True
            else:
                max_info["flat_top"] = False

            # accept peak if it passes threshold check
            if self.threshold_check(array,peak_max,precise_max_height):
                maxima.append(max_info)

        # returns list of dictionary entries containing maxima information
        return maxima

    # finds the left or right deconvolution bound for a given maxima, step = 1 for right bound step = -1 for left bound
    def find_bound(self, array, center, step, frac: float = 0.01, max_width: int = 25):

        nf = self.noise_factor
        max_value = array[center]
        counter = step
        n = len(array)

        # walk along flat-topped peaks
        while(
            abs(counter) <= max_width
            and 0 <= center + counter < n
            and array[center + counter] == max_value
        ):
            counter += step

        # force a bound if plateau is not escaped
        if abs(counter) > max_width or not (0 <= center + counter < n):

            pos = center + step * max_width

            if pos < 0:
                pos = 0
            elif pos >= n:
                pos = n-1

            return pos

        # handle normal peaks
        min_value = array[center + counter]

        # iterate up to 12 setps in given direction from center
        while(abs(counter) <= max_width 
              and 0 <= center + counter < n
        ):

            value = array[center + counter]
            
            # if the value at this step is less than the current min, set the min to this value
            if value < min_value:
                min_value = value

            # if the value at this step is less than frac of max close window here
            if value < frac * max_value:
                return center + counter
                
            
            # if the value at this step is more than 5 nf greater than the minimum close the window at the previous step
            if value > 5 * nf + min_value:
                return center + counter - step
            
            # increment counter
            counter += step

        # if no previous checks returned a value close window at 25 steps from the max
        pos = center + step * max_width

        if pos < 0:
            pos = 0
        elif pos >= n:
            pos = n-1

        return pos

    # finds a quadratic fit for a set of 3 points in an array
    def quadratic_fit(self, array, center):

        # get map to convert scans to minutes
        scan_map = self.spectra_metadata
        # x values for fit, the center index and its two direct neighbors (in minutes)
        x_points = np.array([scan_map[center-1], scan_map[center], scan_map[center+1]])
        # y values for fit, from row corrosponding to x values
        y_points = array[[center-1,center,center+1]]

        # perform quadratic numpy polyfit, returning coefficients in a,b,c for ax^2 + bx + c form
        coeffs = np.polyfit(x_points, y_points, 2)
        a,b,c = coeffs

        # calcluate the precise maxima of the fit
        max_x = -b/(2*a)

        # array of left point(max_x-1), max, and right point (max_x +1)
        x_values = np.array([max_x-1, max_x, max_x+1]).astype(float)
        # array of y values corrosponding to x_values 
        y_values = a*x_values**2 + b*x_values + c

        fit_result = {
            'x_values' : x_values,
            'y_values' : y_values,
            'coeffs' : coeffs
        }

        return fit_result
    
    # checks if peak is above rejection threshold (4 noise units is base but we can adjust in the future)
    def threshold_check(self, row, peak_idx, height):

        threshold = 4 * self.noise_factor * row[peak_idx]**0.5 

        if height < threshold:
            return False
        else:
            return True

    # calculates convolution value for a single peak, used to see if peak is singlet or not
    def convolution_value(self,row,max):
        
        # holds the sum of all rates of sharpness calculated for the peak
        rate_sum = 0

        # value to prevent divide by 0 errors
        eps = 1e-12

        # loop over all the scans in the 3 scan window and calculate rate for each
        for i in range(1,4):
            term1 = (row[max+(i+1)] - row[max+i]) / (row[max+i] + eps)
            term2 = (row[max-(i+1)] - row[max-i]) / (row[max-i] + eps)

            rate_sum += term1+term2

        return rate_sum
    
    # endregion

    # region Baseline Calculation

    # calculates a tentative baseline for a percieved component (minutes not scans)
    def tentative_baseline(self,left_bound,right_bound,array):

        # get scan to minute map and generate the left and right times
        scan_map = self.spectra_metadata
        s = max(scan_map.keys())
        l = min(scan_map.keys())

        # create componenet array
        component_array = array[left_bound:right_bound+1]

        # creates an x-values array to use later for baseline computing (in min, not scans)
        try:
            x = np.array([scan_map[i] for i in range(left_bound,right_bound+1)], dtype=float)
        except Exception:
            print(f"Max key: {s}\n Min key: {l}")
            print(f"Left Bound: {left_bound}    Right Bound: {right_bound}")

        # get the index of the peak maximum
        max_idx = np.argmax(component_array)
        # if peak is flat top then assign the max to the midpoint
        if max_idx == 0 or max_idx == (right_bound - left_bound):
            max_idx = (right_bound - left_bound) // 2 
        
        # get the index values of the minimum on the left and on the right of the max
        left_idx = np.argmin(component_array[:max_idx])
        right_idx = np.argmin(component_array[max_idx:]) + max_idx

        # get the intensity values associated with both these minimums
        left_val = component_array[left_idx]
        right_val = component_array[right_idx]

        # get linear baseline variables
        m = (right_val - left_val) / (right_idx - left_idx)
        b = left_val - m * left_idx

        # generate tentative baseline array
        baseline_array = m * x + b

        # shfit baseline down if any of its values are greater than the value of input array at same index
        diffs = []
        for idx, element in enumerate(baseline_array):
            if element > component_array[idx]:
                diff = element - component_array[idx]
                diffs.append(diff)

        # handle y-int/baseline array correction (shifting up/down)
        if len(diffs) > 0:
            y_correct = max(diffs)
            baseline_array -= y_correct
        else:
            y_correct = 0

        # save and return values
        output = {
            'baseline_array' : baseline_array,
            'slope': m,
            'y_int': b-y_correct,
            'left_bound': left_bound,
            'right_bound': right_bound
        }
        return output

    # endregion

    # region Data Collection

    def closest_peak(self, mz: int, rt: float):
        """
        finds the peak closest to the given retention time (rt) value in a given ion chromatogram for ion mz
        Params:
            mz                      M/Z ion chromatogram to search
            rt                      retention time of the peak of interest
        Returns:
            closest_peak            peak from that ion list that is closest to the supplied RT
        """

        cfg = self.cfg
        threshold = cfg.get("rt_threshold")

        try:
            # get indx value of this m/z chromatogram
            row_idx = self.unique_mzs.index(mz)
            # get the peak list for this row
            peaks = self.peak_list[row_idx]

        except Exception as e:
            print(f"Error locating ion chromatogram for ion: {mz}\n{e}")
            print(f"Unique mzs:\n {self.unique_mzs}")
            return None
        
        # if no peaks are found then store an empty peak
        if len(peaks) == 0:
            closest_peak = {
                'left_bound' : np.nan,
                'right_bound' : np.nan,
                'center' : np.nan,
                'precise_max_location' : np.nan,
                'precise_max_height' : np.nan,
                'max_abundance' : np.nan,
                'bin' : np.nan,
                'conv_value' : np.nan,
                'ion' : np.nan,
                'tentative_baseline': np.nan,
                'width': np.nan,
                'width_flag': np.nan,
                'flat_top': np.nan,
            }
        
        # find the peak closest to specified RT
        try:
            closest_peak = min(
                peaks,
                key = lambda p: abs(p['precise_max_location'] - rt)
            )
        except Exception as e:
            print(f"No peaks availbe in ion chromatogram for ion: {mz}\n{e}")
            print(len(peaks))

        # find rt difference
        if  np.isnan(closest_peak['precise_max_location']):
            diff = np.nan
        else:
            diff = abs(closest_peak['precise_max_location'] - rt)

        # save rt difference to closest peak
        closest_peak["rt_diff"] = diff
        
        # update rt valid flag
        if np.isnan(diff) or diff > threshold:
            closest_peak["rt_valid"] = False
        else:
            closest_peak["rt_valid"] = True
        
        return closest_peak
    
    def integrate_peak(self, peak: dict):
        """
        uses trapazoidal integration to get a peak area value
        Params:
            peak                    dict entry for the peak to be integrated
        """
        # check to see if it is flat-top, if it is do not integrate
        if peak["flat_top"]:
            peak["area"] = 0
            return None
        # check if peak is out of bounds for valid RT, if so then do not integrate
        if not peak["rt_valid"]:
            peak["area"] = 0

        # get symmetry threshold
        cfg = self.cfg
        end_threshold = cfg.get("endpoint_threshold")

        # start and end idx for this peak in intensity matrix
        start = peak['left_bound']
        end = peak["right_bound"]

        # get correct ion chromatogram
        row_idx = self.unique_mzs.index(peak['ion'])
        row = self.intensity_matrix[row_idx]

        # get this peak's abundance array
        signal =  row[start:end+1]

        # check to see how close the endpoints are
        left = signal[0]
        right = signal[-1]
        max_val = signal[peak['center'] - peak["left_bound"]]
        
        # calculate % difference from each endpoint to max
        left_diff = 100 * (max_val - left) / max_val
        right_diff = 100 * (max_val - right) / max_val

        # calculate the diff between endpoints
        symmetry = left_diff - right_diff
        peak["bound_symmetry"] = symmetry
        if abs(symmetry) > end_threshold:
            peak["symmetry_valid"] = False
        else:
            peak["symmetry_valid"] = True

        # adjust to baseline
        if len(signal) != len(peak['tentative_baseline']['baseline_array']):
            print(f"Array length: {len(signal)}\nBaseline length: {len(peak['tentative_baseline'])}")
            raise ValueError("baseline and signal arrays are of different length")
        
        net = signal - peak['tentative_baseline']['baseline_array']

        # tarpazoidal integrate the net value
        peak_area = np.trapezoid(net)
        peak["area"] = peak_area
    
    @staticmethod
    def collect_data(matrices: list, molecules: list, mzs: list, rts: list):
        """
        Collects all peaks from a given list of matrices that corrospond to molecule/mz/rt gropuing specified
        Params:
            matrices                            list of IntensityMatrix objects to parse
            molecules,mzs,rts                   lists (index matched) of moleucle,mz,rt triplets
        Returns:
            output                              dict of sample_name: peak list values
        """
        output = {}

        for matrix in matrices:
            name = matrix.spectra_name
            peaks = []

            for idx,molecule in enumerate(molecules):
                if np.isnan(matrix.noise_factor):
                    print(f"Spectra {matrix.spectra_name} NF error\nNF: {matrix.noise_factor}")
                peak = matrix.closest_peak(mzs[idx],rts[idx])
                peak["molecule"] = molecule
                matrix.integrate_peak(peak)
                peaks.append(peak)

            output[name] = peaks

        return output

    # endregion

    # region Data Visualization

    def width_histogram(self):
        widths = []
        for row in self.peak_list:
            for peak in row:
                width = peak["right_bound"] - peak["left_bound"]
                widths.append(width)
        
        max_width = max(widths)
        min_width = min(widths)

         # bin
        bins = np.arange(
            min_width,
            max_width + 2,
            1
        )
        
        plt.figure(figsize=(8,5))
        plt.hist(widths,bins=bins)
        plt.xlabel("Peak width (scans)")
        plt.ylabel("Count")
        plt.title("Peak Width Distribution")
        plt.tight_layout()
        plt.show()

    def plot_ic(self, mz: int):
        """
        Plots a given m/z ion chromatogram for visualization
        """
        print(f"Noise Factor: {self.noise_factor}")
        row_idx = self.unique_mzs.index(mz)
        row = self.intensity_matrix[row_idx]

        plt.plot(row)
        plt.xlabel("Index")
        plt.ylabel("Abundance")
        plt.title(f"{mz} Ion Chromatogram")
        plt.show()

    # endregion
