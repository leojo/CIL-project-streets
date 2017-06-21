function sLabel = bow_recognition_nearest(histogram, vBoWPos, vBoWNeg)
% Label the histogram by +1 or 0, based on the closest found histogram in
% the positive sampled or negative sampled respectively.
%
% Input
%   histogram       Histogram to label.
%   vBoWPos         All histograms from the positive samples.
%   vBoWNeg         All histograms from the negative samples.
%
% Output:
%   sLabel          1 if the closest neighbour is positive => contains car
%                   0 if the closest neighbour is negative => no car

%% Find the nearest neighbor in the positive and negative sets
[~, distPos] = findnn(histogram, vBoWPos);
[~, distNeg] = findnn(histogram, vBoWNeg);

%% Label the histogram based on the closer neighbor
if distPos < distNeg
    sLabel = 1;
else
    sLabel = 0;
end

end
