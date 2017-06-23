function bow_recognition_multi(histograms, labels, vBoWPos, vBoWNeg, classifierFunction)

image_count = size(histograms,1)
pos = 0;
neg = 0;
predicted = zeros(size(labels));

for i = 1:image_count
    % classify each histogram
    l = classifierFunction(histograms(i,:), vBoWPos, vBoWNeg);
    predicted(i) = l;
    
    % compare the result to the respective label
    if (l == labels(i)) % positive
        pos = pos + 1;
    else %negative
        neg = neg + 1;
    end
end

disp(['Percentage of correctly classified images:' num2str(pos/image_count)]);

end

