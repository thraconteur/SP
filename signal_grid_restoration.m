%% =======================================================================
%  THE SIGNAL GRID -- Image Restoration Pipeline
%  Electrothon | Electronics Club - Signal Processing Domain
%
%  PROBLEM
%  -------
%  Each "cursed_schematic" image has been corrupted in two independent
%  ways:
%    (1) Salt-and-pepper noise         -> isolated black/white pixels
%    (2) A repeating interference grid -> a periodic pattern that shows
%                                          up as sharp, isolated peaks in
%                                          the frequency domain
%  The two corruptions are tackled in the domain where each is easiest to
%  remove: (1) in the SPATIAL domain with a median filter, and (2) in the
%  FREQUENCY domain with a notch filter built from automatically detected
%  FFT peaks.
%
%  This single script is fully adaptive: it loops over every
%  "cursed_schematic_*.png" file it finds next to it, so it works whether
%  it is given 1 test image or 20, with no per-image tuning required and
%  no separate copies of the file needed.
% =========================================================================

clear; clc; close all;

%% ----------------------- Section 0: Settings ---------------------------
% All spatial parameters below are expressed as FRACTIONS of the image
% size (not fixed pixel counts), so the same script generalises correctly
% to images of any resolution, not just the 256x256 samples supplied here.

medianWindow      = [3 3];  % spatial median filter window (salt & pepper)
excludeRadiusFrac = 0.06;   % never touch the notch filter inside this
                            % radius (fraction of image size) around DC
                            % -> protects genuine low-frequency content
bgSmoothFrac      = 0.08;   % window (fraction of image size) used to
                            % estimate the smooth "background" trend of
                            % the spectrum
peakStdThreshold  = 4.0;    % keep a candidate peak only if it stands this
                            % many standard deviations above the LOCAL
                            % background level
notchRadiusFrac   = 0.016;  % radius (fraction of image size) of each
                            % Gaussian notch dip
notchStrength     = 0.97;   % 0 = no attenuation, 1 = fully zeroed
maxPeaksToMask    = 80;     % safety cap so a pathological image can never
                            % cause runaway masking

inputPattern = 'cursed_schematic_*.png';
outputFolder = 'restored_output';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% ---------------- Section 0b: Blueprint colour map (BONUS) -------------
% Classic engineering blueprints are white/cyan linework on a deep blue
% background, so low intensities are mapped near navy blue and high
% intensities are mapped towards white.
nMapLevels   = 256;
t            = linspace(0, 1, nMapLevels)';
blueDark     = [0.03 0.15 0.38];
blueLight    = [0.90 0.97 1.00];
blueprintMap = (1 - t) .* blueDark + t .* blueLight;

%% ------------------------- Main loop over images -----------------------
files = dir(inputPattern);
fprintf('Found %d image(s) matching "%s"\n\n', numel(files), inputPattern);

for k = 1:numel(files)
    fname = files(k).name;
    [~, baseName, ~] = fileparts(fname);
    fprintf('--- Processing %s ---\n', fname);

    %% 1. Load & prepare ---------------------------------------------------
    original = imread(fname);
    if size(original, 3) == 3
        original = rgb2gray(original);   % safety net; inputs are already grayscale
    end
    original = double(original);
    [rows, cols] = size(original);

    %% 2. THE SPATIAL CLEANSING (median filter) -----------------------------
    % WHY: salt-and-pepper noise is a set of isolated, extreme-valued
    %      pixels. A linear smoothing filter (mean/Gaussian) would blur
    %      that noise INTO neighbouring pixels rather than remove it. A
    %      median filter instead replaces each pixel with the median of
    %      its neighbourhood, so an isolated extreme value is simply
    %      outvoted and discarded, while edges and flat regions survive.
    % HOW:  a 3x3 window was tested against 5x5. Both removed the impulse
    %       noise, but 5x5 visibly rounded off the sharp corners of the
    %       schematic shapes, so 3x3 was kept as the better trade-off
    %       between denoising strength and detail preservation.
    spatialCleaned = medfilt2(uint8(original), medianWindow);
    spatialCleaned = double(spatialCleaned);

    %% 3. THE FREQUENCY EXORCISM (2D FFT) ------------------------------------
    % Move to the frequency domain and centre the zero-frequency (DC) term
    % so radial distance from the middle of the image corresponds directly
    % to spatial frequency.
    F           = fft2(spatialCleaned);
    Fshifted    = fftshift(F);
    magSpectrum = log(1 + abs(Fshifted));   % log scale purely for viewing/analysis

    %% 3a. Automated peak detection (BONUS: adapts to any image, ------------
    %%     no hardcoded coordinates)
    % WHY: hardcoding pixel coordinates for the noise peaks would only work
    %      for one exact grid pattern; the sample images already show
    %      different grid orientations and spacings, so a fixed set of
    %      coordinates would not generalise. An adaptive detector is both
    %      more robust and reusable on unseen images.
    % HOW: a real image's spectrum already falls off smoothly from the
    %      centre outward, so comparing every pixel to one single global
    %      threshold would wrongly flag ordinary low-frequency content near
    %      the centre. Instead:
    %        (i)   estimate the smooth local trend of the spectrum with a
    %              wide median filter -- a median filter is robust to
    %              sparse, sharp spikes, so it reports what the spectrum
    %              would look like WITHOUT the interference;
    %        (ii)  subtract that trend to get a residual that is close to
    %              flat everywhere except at the true interference spikes;
    %        (iii) flag a residual pixel as a noise peak only if it is a
    %              local maximum AND stands far above the residual's own
    %              statistics (mean + k*std), AND lies outside a small
    %              protected radius around the DC term.
    minDim        = min(rows, cols);
    excludeRadius = max(6, round(excludeRadiusFrac * minDim));
    bgWin         = round(bgSmoothFrac * minDim);
    if mod(bgWin, 2) == 0
        bgWin = bgWin + 1;             % medfilt2 requires an odd window
    end
    notchRadius = max(2, notchRadiusFrac * minDim);

    background = medfilt2(magSpectrum, [bgWin bgWin]);
    residual   = magSpectrum - background;

    [colGrid, rowGrid] = meshgrid(1:cols, 1:rows);
    centreRow = floor(rows/2) + 1;
    centreCol = floor(cols/2) + 1;
    distFromCentre = sqrt((rowGrid - centreRow).^2 + (colGrid - centreCol).^2);
    outsideCore = distFromCentre > excludeRadius;

    isRegionalMax  = imregionalmax(residual);
    candidatePeaks = isRegionalMax & outsideCore;

    backgroundStats = residual(outsideCore);
    peakThreshold   = mean(backgroundStats) + peakStdThreshold * std(backgroundStats);
    finalPeakMask   = candidatePeaks & (residual > peakThreshold);

    [peakRows, peakCols] = find(finalPeakMask);
    peakVals = residual(sub2ind(size(residual), peakRows, peakCols));
    if numel(peakVals) > maxPeaksToMask
        [~, sortOrder] = sort(peakVals, 'descend');
        sortOrder = sortOrder(1:maxPeaksToMask);
        peakRows  = peakRows(sortOrder);
        peakCols  = peakCols(sortOrder);
    end
    fprintf('   detected %d interference peak(s) automatically\n', numel(peakRows));

    %% 3b. Build the notch filter mask ---------------------------------------
    % WHY a GAUSSIAN notch rather than a hard (ideal) notch: an ideal notch
    % has an abrupt cutoff, which introduces ringing artefacts (the Gibbs
    % phenomenon) into the reconstructed image. A smooth Gaussian dip
    % suppresses the same interference peak but transitions gradually,
    % keeping the reconstructed schematic visually clean.
    notchMask = ones(rows, cols);
    for i = 1:numel(peakRows)
        dip = notchStrength * exp( -((colGrid - peakCols(i)).^2 + (rowGrid - peakRows(i)).^2) ...
                                     / (2 * notchRadius^2) );
        notchMask = notchMask .* (1 - dip);
    end

    %% 4. Apply the mask and invert back to the spatial domain ---------------
    Fmasked       = Fshifted .* notchMask;
    reconstructed = real(ifft2(ifftshift(Fmasked)));

    % Small negative/overshoot values are a normal side effect of notch
    % filtering (mild Gibbs ringing); they are clipped back into the valid
    % intensity range rather than rescaling the whole image, so the true
    % brightness of the schematic is preserved.
    restored = uint8(max(0, min(255, reconstructed)));

    %% 5. BONUS: blueprint colour mapping -------------------------------------
    restoredForColour = mat2gray(reconstructed);          % contrast-stretched, for a punchy look
    restoredIndexed   = gray2ind(restoredForColour, nMapLevels);
    restoredBlueprint = ind2rgb(restoredIndexed, blueprintMap);

    %% 6. Save the required 5-panel comparison figure -------------------------
    fig = figure('Visible', 'off', 'Position', [50 50 1500 350], 'Color', 'w');

    subplot(1,5,1); imshow(uint8(original));       title('1. Original (corrupted)');
    subplot(1,5,2); imshow(uint8(spatialCleaned)); title('2. After median filter');
    subplot(1,5,3); imshow(magSpectrum, []);       title('3. 2D FFT magnitude');
    subplot(1,5,4); imshow(notchMask, []);         title('4. Notch (frequency) mask');
    subplot(1,5,5); imshow(restored);              title('5. Final restored image');

    panelFile = fullfile(outputFolder, [baseName '_panel.png']);
    print(fig, panelFile, '-dpng', '-r150');
    close(fig);

    %% 7. Save the standalone image assets ------------------------------------
    imwrite(restored,          fullfile(outputFolder, [baseName '_restored.png']));
    imwrite(restoredBlueprint, fullfile(outputFolder, [baseName '_blueprint.png']));

    fprintf('   saved: %s, %s, %s\n\n', [baseName '_panel.png'], ...
            [baseName '_restored.png'], [baseName '_blueprint.png']);
end

fprintf('All done. Outputs are in the "%s" folder.\n', outputFolder);
