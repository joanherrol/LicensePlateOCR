%% SHORT PROJECT
% Joan Hervás and Óscar Estudillo

%% Goal
% The goal of this project is to implement code for the detection and
% reading of license plates in an image dataset of cars in exterior
% scenarios.

%% Feature dictionaries
digitsDictionary = createDigitDictionary();
lettersDictionary = createLetterDictionary();

%% Train a classifier 
digitsClassifier = createDigitClassifier();
lettersClassifier = createLetterClassifier();

%% Process Images
good = 0;
notfound = 0;
nocandidates = 0;

notfoundnames = string([]);
nocandidatesnames = string([]);

close all
%a = dir('VC SP QT 2022/day_color(small sample)\*.jpg');
a = dir('day_color(small sample)\*.jpg');
nf = size(a);
figure
for i = 1:nf 
    %Read image
    filename = horzcat(a(i).folder,'/',a(i).name);
    name = a(i).name;
    I = imread(filename);
    [~, w] = size(I);
    gray = rgb2gray(I); 
     
    %Preprocess Image
    bin = platebin(gray);
    
    %Extract Plate Candidates
    candidates = platecandidates(bin, w);
    [sz, ~] = size(candidates);
    if sz == 0
        bin = hardlightplatebin(gray);
        candidates = platecandidates(bin, w);
    end
    
    %Extract Digits
    [sz, ~] = size(candidates);
    if sz > 0
    
    [plate, digits] = lookforplate(gray, candidates, name);
    
    if length(digits) == 7 && isempty(plate) == 0
        %Extract Plate Features
        features = extractPlateHOGFeatures(I, plate, normalizeDigits(digits));
    
        %Classify Plate Features
        stringPlate = findAndPrintPlateStringClassifier(digitsClassifier, lettersClassifier, features);
    
        %Print result
        printImage(I, name, plate, normalizeDigits(digits), stringPlate);
        good = good + 1;
    else
        imshow(I), title(name + " has plate candidates");
        hold on
        for j = 1:sz
             bbox = candidates(j, :, :, :);
             rectangle('Position',bbox,'EdgeColor','y')
        end
        notfound = notfound + 1;
        notfoundnames = [notfoundnames, name];
    end
    
    else
        imshow(I), title(name + " has no plate candidates");
        nocandidates = nocandidates + 1;
        nocandidatesnames = [nocandidatesnames, name];
    end
    pause
end 
 
accuracy = good / nf(1);

%% Preprocessing Functions
function res = platebin(im)
    [h, w] = size(im);
    clearborder = imclearborder(im);
    norm = mat2gray(clearborder);
    bin = imbinarize(norm);
    area = h*w;
    min = 0.001*area;
    max = 0.15*area;
    filtered = bwareafilt(bin, [min max]);
    mark = imopen(filtered, strel('disk', 5));
    rec = imreconstruct(mark, bin);
    res = imerode(rec, strel('disk', round(w*0.0005)));
end

function res = hardlightplatebin(im)
    [h, w] = size(im);
    clearborder = imclearborder(imflatfield(im, 30));
    bin = imbinarize(clearborder,'adaptive', 'Sensitivity', 0.3);
    area = h*w;
    min = 0.005*area;
    max = 0.15*area;
    filtered = bwareafilt(bin, [min max]);
    mark = imopen(filtered, strel('disk', 5));
    rec = imreconstruct(mark, filtered);
    res = imerode(rec, strel('disk', round(w*0.001)));
end


%% Candidates Functions
function candidates = platecandidates(bin, w)
    [labels, ~] = bwlabel(bin);
    props = regionprops(labels, 'BoundingBox');
    bboxs = cat(1, props.BoundingBox);
    ratios = boundingboxratios(bboxs);
    widths = boundingboxwidths(bboxs);
    condition = ratios >= 2.5 & ratios <= 6 & widths > 0.03*w & widths < 0.14*w;
    candidates = bboxs(condition, :, :, :);
end

function ratios = boundingboxratios(bboxs)
    n = size(bboxs);
    ratios = zeros(n(1), 1);
    for i = 1:n
        bbox = bboxs(i, :, :, :);
        ratio = bbox(3) / bbox(4);
        ratios(i) = ratio;
    end
end

function widths = boundingboxwidths(bboxs)
    n = size(bboxs);
    widths = zeros(n(1), 1);
    for i = 1:n
        bbox = bboxs(i, :, :, :);
        width = bbox(3);
        widths(i) = width;
    end
end

function heights = boundingboxheights(bboxs)
    n = size(bboxs);
    heights = zeros(n(1), 1);
    for i = 1:n
        bbox = bboxs(i, :, :, :);
        height = bbox(4);
        heights(i) = height;
    end
end

function digits = platedigits(plate)
    [h, w] = size(plate);
    plate = imcomplement(mat2gray(plate));
    plate = imadjust(plate);
    plate = imbinarize(plate);
    bborder = blackborder(plate, int16(round(0.28*h)), int16(round(0.07*w)));
    plate = imreconstruct(bborder, plate);
    plate = blackborder(plate, int16(round(0.125*h)), int16(round(0.035*w)));
    platearea = h*w;
    min = 0.005*platearea;
    max = 0.1*platearea;
    filtered = bwareafilt(plate, [min max]);
    filtered = bwareafilt(filtered, 7);
    filtered = imreconstruct(filtered, plate);
    [labels, ~] = bwlabel(filtered, 8);
    props = regionprops(labels, 'BoundingBox');
    bboxs = cat(1, props.BoundingBox);
    heights = boundingboxheights(bboxs);
    widths = boundingboxwidths(bboxs);
    condition = heights >= 0.2*h & widths <= 0.3*w;
    dcandidates = bboxs(condition, :);
    [sz, ~] = size(dcandidates);
    if sz >= 7
        digits = dcandidates;
    else
        [labels, ~] = bwlabel(plate, 8);
        props = regionprops(labels, 'BoundingBox');
        bboxs = cat(1, props.BoundingBox);
        ratios = boundingboxratios(bboxs);
        heights = boundingboxheights(bboxs);
        widths = boundingboxwidths(bboxs);
        condition = heights >= 0.2*h & ratios <= 1.2 & widths <= 0.3*w;
        digits = bboxs(condition, :);
    end
end

function result = isaplate(plate)
    digits = platedigits(plate);
    [sz, ~] = size(digits);
    if sz >= 7
        result = true;
    else 
        result = false;
    end
end

function [plateRes, digitsRes] = lookforplate(gray, candidates, name)
    [sz, ~] = size(candidates);
    plateRes = zeros(0);
    digitsRes = zeros(0);
    if sz > 0
        for i = 1:sz
            bbox = candidates(i, :, :, :);
            plate = imcrop(gray, bbox);
            if isaplate(plate)
                digits = platedigits(plate);
                digitsRes=digits;
                plateRes=bbox;
                break
            end
        end
    end
end

function im = blackborder(im, height, width)
    [n, m] = size(im);

    im(1:1+double(height), :) = 0;
    im(double(n-height):n, :) = 0;
    im(:, 1:1+double(width)) = 0;
    im(:, double(m-width):m) = 0;
end


%% Classifier functions
function res = createDigitClassifier()
    v = ["1","2","3","4","5","6","7","8","9","0"];
    trainingLabels = repelem(v,2);

    % Read image with all the digits
    I1 = rgb2gray(imread("Greek-License-Plate-Font-2004.svg.png"));
    I2 = rgb2gray(imread("Greek-License-Plate-Font-old.jpg"));
    bw1 = ~imbinarize(I1);
    bw2 = ~imbinarize(I2);
    
    numbers1 = regionprops(bw1,'BoundingBox');
    numbers2 = regionprops(bw2,'BoundingBox');

    trainingFeatures = zeros(20, 1080, 'single');

    thisBB1 = normalizeBoundingBox(numbers1(1).BoundingBox);
    thisBB2 = normalizeBoundingBox(numbers2(1).BoundingBox);
    crop1 = imcrop(bw1, thisBB1);
    crop2 = imcrop(bw2, thisBB2);
    [h1, w1] = size(crop1);
    [h2, w2] = size(crop2);
    crop1 = blackborder(crop1, 0, int16(round(0.1*w1)));
    crop2 = blackborder(crop2, 0, int16(round(0.3*w2)));
    cropWithPadding1 = padarray(crop1, [round(0.3*h1),round(0.3*w1)], 0, "both");
    cropWithPadding2 = padarray(crop2, [round(0.3*h2),round(0.3*w2)], 0, "both");
    resized1 = imresize(cropWithPadding1, [120, 100], 'nearest');
    resized2 = imresize(cropWithPadding2, [120, 100], 'nearest');

    [featureVector1, ~] = extractHOGFeatures(resized1, 'CellSize',[16 16]);
    [featureVector2, ~] = extractHOGFeatures(resized2, 'CellSize',[16 16]);
   
  
    trainingFeatures(1, :) = featureVector1;  
    trainingFeatures(2, :) = featureVector2;

    
    j = 3;
    for i = 2 : 10
        thisBB1 = normalizeBoundingBox(numbers1(i).BoundingBox);
        thisBB2 = normalizeBoundingBox(numbers2(i).BoundingBox);
        crop1 = imcrop(bw1, thisBB1);
        crop2 = imcrop(bw2, thisBB2);
        [h1, w1] = size(crop1);
        [h2, w2] = size(crop2);
        cropWithPadding1 = padarray(crop1, [round(0.3*h1),round(0.3*w1)], 0, "both");
        cropWithPadding2 = padarray(crop2, [round(0.3*h2),round(0.3*w2)], 0, "both");
        resized1 = imresize(cropWithPadding1, [120, 100], 'nearest');
        resized2 = imresize(cropWithPadding2, [120, 100], 'nearest');

        [featureVector1, ~] = extractHOGFeatures(resized1, 'CellSize',[16 16]);
        [featureVector2, ~] = extractHOGFeatures(resized2, 'CellSize',[16 16]);



        trainingFeatures(j, :) = featureVector1;  
        trainingFeatures(j+1, :) = featureVector2;

        j = j+2;
    end

    res = fitcknn(trainingFeatures, trainingLabels);
end

function res = createLetterClassifier()
    
    v = [ "A", "B", "E", "H", "I", "K", "M","N","P","T","X","Y","Z"];
    trainingLabels = repelem(v,2);

    % Read image with all the digits
    I1 = rgb2gray(imread("Greek-License-Plate-Font-2004.svg.png"));
    I2 = rgb2gray(imread("Greek-License-Plate-Font-old.jpg"));
    bw1 = ~imbinarize(I1);
    bw2 = ~imbinarize(I2);
    
    numbers1 = regionprops(bw1,'BoundingBox');
    numbers2 = regionprops(bw2,'BoundingBox');

    trainingFeatures = zeros(26, 1080, 'single');
    j=1;
    for i = 1 : 4
        thisBB1 = normalizeBoundingBox(numbers1(i+11).BoundingBox);
        thisBB2 = normalizeBoundingBox(numbers2(i+11).BoundingBox);
        crop1 = imcrop(bw1, thisBB1);
        crop2 = imcrop(bw2, thisBB2);
        [h1, w1] = size(crop1);
        [h2, w2] = size(crop2);
        cropWithPadding1 = padarray(crop1, [round(0.3*h1),round(0.3*w1)], 0, "both");
        cropWithPadding2 = padarray(crop2, [round(0.3*h2),round(0.3*w2)], 0, "both");
        resized1 = imresize(cropWithPadding1, [120, 100], 'nearest');
        resized2 = imresize(cropWithPadding2, [120, 100], 'nearest');

        %pause
        [featureVector1, ~] = extractHOGFeatures(resized1, 'CellSize',[16 16]);
        [featureVector2, ~] = extractHOGFeatures(resized2, 'CellSize',[16 16]);


        trainingFeatures(j, :) = featureVector1;  
        trainingFeatures(j+1, :) = featureVector2;    
        j = j+2;
    end

    thisBB1 = normalizeBoundingBox(numbers1(5+11).BoundingBox);
    thisBB2 = normalizeBoundingBox(numbers2(5+11).BoundingBox);
    crop1 = imcrop(bw1, thisBB1);
    crop2 = imcrop(bw2, thisBB2);
    crop1 = blackborder(crop1, 0, int16(round(0.3*w1)));
    crop2 = blackborder(crop2, 0, int16(round(0.3*w2)));
    [h1, w1] = size(crop1);
    [h2, w2] = size(crop2);
    cropWithPadding1 = padarray(crop1, [round(0.3*h1),round(0.3*w1)], 0, "both");
    cropWithPadding2 = padarray(crop2, [round(0.3*h2),round(0.3*w2)], 0, "both");
    resized1 = imresize(cropWithPadding1, [120, 100], 'nearest');
    resized2 = imresize(cropWithPadding2, [120, 100], 'nearest');

    [featureVector1, ~] = extractHOGFeatures(resized1, 'CellSize',[16 16]);
    [featureVector2, ~] = extractHOGFeatures(resized2, 'CellSize',[16 16]);

    trainingFeatures(j, :) = featureVector1;  
    trainingFeatures(j+1, :) = featureVector2;  
    j = j+2;
    
    for i = 6 : 13
        thisBB1 = normalizeBoundingBox(numbers1(i+11).BoundingBox);
        thisBB2 = normalizeBoundingBox(numbers2(i+11).BoundingBox);
        crop1 = imcrop(bw1, thisBB1);
        crop2 = imcrop(bw2, thisBB2);
        crop1 = bwareafilt(crop1, 1);
        crop2 = bwareafilt(crop2, 1);
        [h1, w1] = size(crop1);
        [h2, w2] = size(crop2);
        cropWithPadding1 = padarray(crop1, [round(0.3*h1),round(0.3*w1)], 0, "both");
        cropWithPadding2 = padarray(crop2, [round(0.3*h2),round(0.3*w2)], 0, "both");
        resized1 = imresize(cropWithPadding1, [120, 100], 'nearest');
        resized2 = imresize(cropWithPadding2, [120, 100], 'nearest');

        [featureVector1, ~] = extractHOGFeatures(resized1, 'CellSize',[16 16]);
        [featureVector2, ~] = extractHOGFeatures(resized2, 'CellSize',[16 16]);


        trainingFeatures(j, :) = featureVector1;  
        trainingFeatures(j+1, :) = featureVector2;  
        j = j+2;
    end
   

    res = fitcknn(trainingFeatures, trainingLabels);
end



function res = extractPlateHOGFeatures(I, plate, digits)

    digitsFeatures = zeros(length(digits), 1080, 'single');
    for k = 1 : length(digits)
           digit = normalizeBoundingBox(digits(k, :));
           digit(1) = digit(1) + plate(1);
           digit(2) = digit(2) + plate(2);
           crop = imcrop(I, digit);
        
           grayIm=(255 - im2gray(crop+100));

           bw=imbinarize(grayIm);
           [h, w] = size(bw);
           cropWithPadding = padarray(bw, [round(0.3*h),round(0.3*w)], 0, "both");
           
           
           resized = imresize(cropWithPadding, [120, 100], 'nearest');
           resized = bwareafilt(resized, 1);

           [features,~] = extractHOGFeatures(resized, 'CellSize',[16 16]);
           digitsFeatures(k, :) = features;
    end
    res = digitsFeatures;
end


function res = findAndPrintPlateStringClassifier(digitClassifier, letterClassifier, digitsFeatures)
    plateString = "";
    %Letters
    for i = 1:3
        predictedLabel = predict(letterClassifier, digitsFeatures(i, :));
        plateString = strcat(plateString, predictedLabel);
    end
    plateString = strcat(plateString, "-");
    %Digits
    for i = 4:7
        predictedLabel = predict(digitClassifier, digitsFeatures(i, :));
        plateString = strcat(plateString, predictedLabel);
    end

    res = plateString;
end

function normalized = normalizeBoundingBox(bbox)
        objratio = 2;
        width = bbox(3);
        height = bbox(4);
        ratio = height / width;

        if (ratio > objratio) %width needs to increase
            newheight = height;
            newwidth = round(height/objratio);
        else  %height needs to increase
            newheight = objratio*width;
            newwidth = width;
        end

        xCoord = bbox(1);
        yCoord = bbox(2);

        heightdif = round((newheight - height)/2);
        widthdif = round((newwidth - width)/2);

        xCoord = xCoord - widthdif;
        yCoord = yCoord - heightdif;

        normalized(1) = xCoord;
        normalized(2) = yCoord;
        normalized(3) = newwidth;
        normalized(4) = newheight;
end

function normalized = normalizeDigits(digits)
        mindistance = 10000;
        maxheight = 0;
        objratio = 2;
        normalized = digits;
        for k = 1 : length(digits)
           digit = digits(k, :);
           height = digit(4);
           width = digit(3);
           ratio = height/width;
           dif = abs(objratio-ratio);
           if (dif < mindistance && height > maxheight)
               mindistance = dif;
               maxheight = height;
           end
        end

        maxwidth = round(height/objratio);

        maxwidth = maxwidth + round(0.4*maxwidth);
        maxheight = maxheight + round(0.2*maxheight);

        for k = 1 : length(digits)
            digit = digits(k, :);
            xCoord = digit(1);
            yCoord = digit(2);
            height = digit(4);
            width = digit(3);

            heightdif = round((maxheight - height)/2);
            widthdif = round((maxwidth - width)/1.75);
    
            xCoord = xCoord - widthdif;
            yCoord = yCoord - heightdif;
    
            digit(1) = xCoord;
            digit(2) = yCoord;
            digit(3) = maxwidth;
            digit(4) = maxheight;

            normalized(k, :) = digit;
        end     
end

%% Feature match functions (discarded)
function res = createDigitDictionary()
    numbersString = ["1","2","3","4","5","6","7","8","9","0"];
    I = rgb2gray(imread("Greek-License-Plate-Font-old.jpg"));
    bw = ~imbinarize(I);
    numbers = regionprops(bw,'BoundingBox');
    plateDictionary = containers.Map('KeyType','char','ValueType','any');
    
    for k = 1 : 10
      thisBB = numbers(k).BoundingBox;
      crop = imcrop(bw, thisBB);
      cropWithPadding = padarray(crop, [20,20], 0, "both");
      corners = detectSIFTFeatures(cropWithPadding);
      [features,~] = extractFeatures(cropWithPadding,corners);
      hold on;
      plateDictionary(numbersString(k)) = features;
    end
    
    res = plateDictionary;
end

function res = createLetterDictionary()
    numbersString = ["A","B","E","H","I","K","M","N","P","T","X","Y","Z"];
    I = rgb2gray(imread("Greek-License-Plate-Font-old.jpg"));
    bw = ~imbinarize(I);
    numbers = regionprops(bw,'BoundingBox');
    plateDictionary = containers.Map('KeyType','char','ValueType','any');
    
    for k = 12 : 24
      thisBB = numbers(k).BoundingBox;
      crop = imcrop(bw, thisBB);
      cropWithPadding = padarray(crop, [20,20], 0, "both");
      corners = detectSIFTFeatures(cropWithPadding);
      [features,~] = extractFeatures(cropWithPadding,corners);
      hold on;
      plateDictionary(numbersString(k-11)) = features;
    end

    res = plateDictionary;
end

function res = extractPlateFeatures(I, plate, digits)
    digitsFeatures = containers.Map('KeyType','double','ValueType','any');
    for k = 1 : length(digits)
           digit = digits(k, :);
           digit(1) = digit(1) + plate(1);
           digit(2) = digit(2) + plate(2);
           crop = imcrop(I, digit);
    
           grayIm=(255 - im2gray(crop+100));
           bw=imbinarize(grayIm);
    
           cropWithPadding = padarray(bw, [20,20], 0, "both");
    
           corners = detectSIFTFeatures(cropWithPadding);
           [features,~] = extractFeatures(cropWithPadding,corners);

    
           digitsFeatures(k, :) = features;

    end
    res = digitsFeatures;
end

function res = findAndPrintPlateString(digitDictionary, letterDictionary, digitsFeatures)
    digitsString = ["1","2","3","4","5","6","7","8","9","0"];
    lettersString= ["A","B","E","H","I","K","M","N","P","T","X","Y","Z"];
    strongestMatch = cell(7,1);
    plateString = "";
    
    %Digits
    for k = 1 : 3
        maxMatchedFeatures = 0;
        %minMetricDistance = 100000000;
        for l = 1 : letterDictionary.length
            [indexPairs, matchMetric] = matchFeatures(digitsFeatures(k), letterDictionary(lettersString(l)));
            n = length(indexPairs);
            if (n > maxMatchedFeatures)
                %distance = sum(matchMetric)/n;
                %if (distance < minMetricDistance)
                    maxMatchedFeatures = n;
                    %minMetricDistance = distance;
                    strongestMatch{k} = lettersString(l);
                %end
            end
        end
        if (isempty(strongestMatch{k}))
            strongestMatch{k,1}="*";
        end
        plateString = strcat(plateString, strongestMatch{k});
    end
   
    plateString = strcat(plateString, "-");

    for k = 4 : 7
        maxMatchedFeatures = 0;
        %minMetricDistance = 100000000;
        for l = 1 : digitDictionary.length
            [indexPairs, matchMetric] = matchFeatures(digitsFeatures(k), digitDictionary(digitsString(l)));
            n = length(indexPairs);
            if (n > maxMatchedFeatures)
                %distance = sum(matchMetric)/n;
                %if (distance < minMetricDistance)
                    maxMatchedFeatures = n;
                    %minMetricDistance = distance;
                    strongestMatch{k} = digitsString(l);
                %end
            end
        end
        if (isempty(strongestMatch{k}))
            strongestMatch{k}="*";
        end
        plateString = strcat(plateString, strongestMatch{k});
    end
    res = plateString;
end

function printImage(I, name, plate, digits, plateString)
    [n, ~] = size(digits);
    imshow(I), title(name + " has plate: " + plateString);
    hold on
    rectangle('Position',plate,'EdgeColor','y')
    for j = 1:n
        digit = digits(j, :, :, :);
        digit(1) = digit(1) + plate(1);
        digit(2) = digit(2) + plate(2);
        rectangle('Position',digit,'EdgeColor','g')
    end
    hold off
end





