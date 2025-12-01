function visualize_flex_contacts_pressure(matFile, outputVideo, maxFrames, targetFlex, gridTheta, gridAxial, smoothWindowSec, gaussianSigmaCells, showScatter, maxContacts)
%VISUALIZE_FLEX_CONTACTS_PRESSURE Animate contact forces as a pressure map.
%   visualize_flex_contacts_pressure()                         % uses flex_contacts_simple.mat
%   visualize_flex_contacts_pressure('file.mat','out.mp4')    % save MP4
%   visualize_flex_contacts_pressure('file.mat','',120,[],[],[],0.03,1.2) % Gaussian blur
%   visualize_flex_contacts_pressure('file.mat','',120,[],[],[],0.03,1.2,false,4e5) % cap contacts
%
% Renders |force_world| magnitudes onto a (theta, axial) grid per frame,
% giving a heatmap/pressure look. Optional per-position temporal smoothing
% (moving average) and spatial Gaussian blur for softer blobs.

if nargin < 1 || isempty(matFile),   matFile   = 'flex_contacts_simple.mat'; end
if nargin < 2,                       outputVideo = []; end
if nargin < 3 || isempty(maxFrames), maxFrames = 240; end
if nargin < 4,                       targetFlex = []; end
if nargin < 5 || isempty(gridTheta), gridTheta = 120; end
if nargin < 6 || isempty(gridAxial), gridAxial = 80; end
if nargin < 7 || isempty(smoothWindowSec), smoothWindowSec = 0.03; end
if nargin < 8 || isempty(gaussianSigmaCells), gaussianSigmaCells = 1.2; end
if nargin < 9 || isempty(showScatter), showScatter = false; end
if nargin < 10 || isempty(maxContacts), maxContacts = 4e5; end

S = load(matFile);
for f = ["time","flex_id","pos","force_world","normal"]
    if ~isfield(S, f)
        error("Field '%s' missing in %s", f, matFile);
    end
end

time = S.time(:);
pos = double(S.pos);
if isfield(S, 'pos_local') && ~isempty(S.pos_local)
    pos_local = double(S.pos_local);
else
    pos_local = pos;
end
forceW = double(S.force_world);
mag = vecnorm(forceW, 2, 2);
fid = S.flex_id(:);

% Optional flex filter
if ~isempty(targetFlex)
    maskFlex = fid == targetFlex;
    if ~any(maskFlex)
        error('No contacts found for flex_id == %d', targetFlex);
    end
    time       = time(maskFlex);
    pos        = pos(maskFlex, :);
    pos_local  = pos_local(maskFlex, :);
    forceW     = forceW(maskFlex, :);
    mag        = mag(maskFlex, :);
    fid        = fid(maskFlex);
end

% Optional contact cap to speed up preprocessing
if maxContacts > 0 && numel(time) > maxContacts
    stride = ceil(numel(time) / maxContacts);
    keepIdx = 1:stride:numel(time);
    time       = time(keepIdx);
    pos        = pos(keepIdx, :);
    pos_local  = pos_local(keepIdx, :);
    forceW     = forceW(keepIdx, :);
    mag        = mag(keepIdx, :);
    fid        = fid(keepIdx);
    fprintf('Downsampled contacts: stride %d (kept %d of %d)\n', stride, numel(time), numel(S.time));
end

% Sort by time
[time, order] = sort(time);
pos        = pos(order, :);
pos_local  = pos_local(order, :);
mag        = mag(order, :);
fid        = fid(order);

% Estimate timestep from unique sample times (before any downsampling)
tUniqueAll = unique(time);
if numel(tUniqueAll) >= 2
    dtEstimate = median(diff(tUniqueAll));
else
    dtEstimate = median(diff(time));
end
if ~isfinite(dtEstimate) || dtEstimate <= 0
    dtEstimate = 1/60; % fallback to something reasonable
end

% Unwrap positions to cylindrical coordinates using a known body axis (per flex)
thetaAll = nan(size(time));
axialAll = nan(size(time));
flexVals = unique(fid(:)).';
axisVecKnown = [0; 0; 1]; % body +Z as axial
refVecKnown = [1; 0; 0];  % reference direction in body frame for theta=0
for flexVal = flexVals
    maskFlex = fid == flexVal;
    if nnz(maskFlex) == 0
        continue;
    end
    rel = pos_local(maskFlex, :);
    axisVec = axisVecKnown;
    axisVec = axisVec / norm(axisVec);
    axial = rel * axisVec;
    radial = rel - axial * axisVec.';
    radialNorm = vecnorm(radial, 2, 2);
    valid = radialNorm > 1e-12;
    if ~any(valid)
        axialAll(maskFlex) = axial;
        thetaAll(maskFlex) = 0;
        continue;
    end
    refVec = refVecKnown - axisVec.' * refVecKnown * axisVec; % ensure orthogonal
    if norm(refVec) < 1e-12
        refVec = radial(find(valid, 1), :) / radialNorm(find(valid, 1));
    else
        refVec = refVec / norm(refVec);
    end
    binormal = cross(axisVec, refVec);
    if ~any(binormal)
        binormal = [0; 1; 0];
    end
    binormal = binormal / norm(binormal);

    theta = zeros(size(axial));
    theta(valid) = atan2(radial(valid, :) * binormal, radial(valid, :) * refVec);
    theta = mod(theta, 2*pi);

    thetaAll(maskFlex) = theta;
    axialAll(maskFlex) = axial;
end

% Downsample unique times for playback speed
tUnique = tUniqueAll;
if numel(tUnique) > maxFrames
    keepIdx = round(linspace(1, numel(tUnique), maxFrames));
    tUnique = tUnique(keepIdx);
end
nFrames = numel(tUnique);

fprintf('Animating %d frames from %s\n', nFrames, matFile);

% Fixed axes and grid
thetaLo = 0; thetaHi = 2*pi;
validAx = axialAll(~isnan(axialAll));
if isempty(validAx)
    axMin = -0.5; axMax = 0.5;
else
    axMin = min(validAx);
    axMax = max(validAx);
end
axRange = axMax - axMin;
if ~isfinite(axRange) || axRange <= 0
    axRange = 1e-3;
end
pad = 0.05 * axRange;
axLo = axMin - pad;
axHi = axMax + pad;

thetaEdges = linspace(thetaLo, thetaHi, gridTheta + 1);
axialEdges = linspace(axLo, axHi, gridAxial + 1);

% Gaussian kernel for spatial blur (applied as weighted average)
if gaussianSigmaCells > 0
    rad = ceil(3 * gaussianSigmaCells);
    [gx, gy] = ndgrid(-rad:rad, -rad:rad);
    gKernel = exp(-(gx.^2 + gy.^2) / (2 * gaussianSigmaCells^2));
    gKernel = gKernel / sum(gKernel(:));
else
    gKernel = [];
end

magMax = max(mag);
if magMax <= 0
    magMax = 1; % avoid zero clim
end

fig = figure('Name', 'Flex contact pressure');
ax  = axes('Parent', fig);
hold(ax, 'on');
grid(ax, 'on');
set(ax, 'YDir', 'normal');
xlabel(ax, 'theta (rad)'); ylabel(ax, 'axial (m)');
title(ax, 't = 0.000 s');
cb = colorbar(ax); ylabel(cb, '|force| (N)');
xlim(ax, [thetaLo, thetaHi]);
ylim(ax, [axLo, axHi]);
set(ax, 'XLimMode', 'manual', 'YLimMode', 'manual');
axis(ax, 'manual');
colormap(ax, parula);

blankGrid = zeros(gridAxial, gridTheta);
pressureImg = imagesc([thetaLo thetaHi], [axLo axHi], blankGrid, 'Parent', ax);
alphaData = zeros(size(blankGrid));
set(pressureImg, 'AlphaData', alphaData);

sc = [];
if showScatter
    mask0 = time == tUnique(1);
    sc = scatter(ax, thetaAll(mask0), axialAll(mask0), 10, mag(mask0), 'filled', 'MarkerEdgeColor', 'k');
end

writer = [];
if ~isempty(outputVideo)
    writer = VideoWriter(outputVideo, 'MPEG-4');
    writer.FrameRate = 60;
    open(writer);
end

for k = 1:nFrames
    mask = time == tUnique(k);
    thetaNow = thetaAll(mask);
    axialNow = axialAll(mask);
    magNow = mag(mask);

    tBin = discretize(thetaNow, thetaEdges);
    aBin = discretize(axialNow, axialEdges);
    validBin = ~isnan(tBin) & ~isnan(aBin);
    tBin = tBin(validBin);
    aBin = aBin(validBin);
    magBin = magNow(validBin);

    if isempty(tBin)
        gridVals = zeros(gridAxial, gridTheta);
        countGrid = gridVals;
    else
        linIdx = sub2ind([gridAxial, gridTheta], aBin, tBin);
        gridVals = accumarray(linIdx, magBin, [gridAxial * gridTheta, 1], @max, NaN);
        gridVals = reshape(gridVals, [gridAxial, gridTheta]);
        countGrid = accumarray(linIdx, 1, [gridAxial * gridTheta, 1], @sum, 0);
        countGrid = reshape(countGrid, [gridAxial, gridTheta]);
    end

    % Apply spatial smoothing as a weighted average
    if ~isempty(gKernel)
        vals0 = gridVals;
        vals0(~isfinite(vals0)) = 0;
        smVals = conv2(vals0, gKernel, 'same');
        smCnt = conv2(countGrid, gKernel, 'same');
        gridVals = smVals ./ max(smCnt, eps);
        countGrid = smCnt;
    end

    alphaData = min(countGrid ./ max(countGrid(:) + eps), 1); %#ok<NASGU>
    set(pressureImg, 'CData', gridVals, 'AlphaData', alphaData);
    set(ax, 'CLim', [0 magMax]);
    title(ax, sprintf('t = %.3f s', tUnique(k)));

    if showScatter && ~isempty(sc)
        set(sc, 'XData', thetaNow, 'YData', axialNow, 'CData', magNow);
    end

    drawnow;
    if ~isempty(writer)
        writeVideo(writer, getframe(fig));
    end
end

if ~isempty(writer)
    close(writer);
    fprintf('Saved %s\n', outputVideo);
end
end
