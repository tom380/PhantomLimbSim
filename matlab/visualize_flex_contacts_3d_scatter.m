function visualize_flex_contacts_3d_scatter(matFile, outputVideo, maxFrames, targetFlex, smoothWindowSec, maxContacts, sizeScale, splitFrontBack, syncVideo)
%VISUALIZE_FLEX_CONTACTS_3D_SCATTER Unwrapped scatter with size = |force|.
%   visualize_flex_contacts_3d_scatter()                     % uses flex_contacts_simple.mat
%   visualize_flex_contacts_3d_scatter('file.mat','out.mp4') % save MP4
%   visualize_flex_contacts_3d_scatter('file.mat','',240,[],0.03,4e5,1.0,true)
%   visualize_flex_contacts_3d_scatter('file.mat','',240,[],0.03,4e5,1.0,true,'video.mp4') % synced video panel
%
% - Unwraps to (theta, axial) per flex (body +Z axis as axial).
% - Point size scales with smoothed |force|.
% - Temporal smoothing per contact location to reduce flicker.

if nargin < 1 || isempty(matFile),   matFile   = 'flex_contacts_simple.mat'; end
if nargin < 2,                       outputVideo = []; end
if nargin < 3 || isempty(maxFrames), maxFrames = 240; end
if nargin < 4,                       targetFlex = []; end
if nargin < 5 || isempty(smoothWindowSec), smoothWindowSec = 0.03; end
if nargin < 6 || isempty(maxContacts), maxContacts = 4e5; end
if nargin < 7 || isempty(sizeScale), sizeScale = 1.0; end
if nargin < 8 || isempty(splitFrontBack), splitFrontBack = false; end
if nargin < 9, syncVideo = []; end

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

% Estimate timestep
tUniqueAll = unique(time);
if numel(tUniqueAll) >= 2
    dtEstimate = median(diff(tUniqueAll));
else
    dtEstimate = median(diff(time));
end
if ~isfinite(dtEstimate) || dtEstimate <= 0
    dtEstimate = 1/60;
end

% Smooth |force| per contact location to reduce flicker
magSm = mag;
if smoothWindowSec > 0
    winSamples = max(1, round(smoothWindowSec / dtEstimate));
    if winSamples > 1
        posTol = 1e-5;
        posKey = round(pos_local / posTol) * posTol;
        [~, ~, posIdx] = unique(posKey, 'rows', 'stable');
        for pid = 1:max(posIdx)
            rows = posIdx == pid;
            if nnz(rows) <= 1
                continue;
            end
            magSm(rows) = movmean(mag(rows), winSamples, 'Endpoints', 'shrink');
        end
    end
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
    axisVec = axisVecKnown / norm(axisVecKnown);
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

% Fixed axes
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

fig = figure('Name', 'Flex contact scatter (size = |force|)');
ax = [];
axVideo = [];
vidObj = [];
vidFrame = [];
vidFrameIdx = 1;
vidTotalFrames = [];
vidFPS = [];
hasVideo = ~isempty(syncVideo);
if hasVideo
    vidObj = VideoReader(syncVideo);
    vidFPS = vidObj.FrameRate;
    vidTotalFrames = ceil(vidObj.Duration * vidFPS);
    if hasFrame(vidObj)
        vidFrame = readFrame(vidObj);
    end
end

if splitFrontBack
    if hasVideo
        tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        axVideo = nexttile(tl, 1);
        axFront = nexttile(tl, 2);
        axBack  = nexttile(tl, 3);
        axs = [axFront, axBack];
    else
        tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        axFront = nexttile(tl, 1);
        axBack  = nexttile(tl, 2);
        axs = [axFront, axBack];
    end
else
    if hasVideo
        tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
        axVideo = nexttile(tl, 1);
        ax = nexttile(tl, 2);
    else
        ax = axes('Parent', fig);
    end
end

magMax = max(magSm);
if magMax <= 0 || ~isfinite(magMax)
    magMax = 1;
end
sMin = 8 * sizeScale;
sMax = 60 * sizeScale;

writer = [];
if ~isempty(outputVideo)
    writer = VideoWriter(outputVideo, 'MPEG-4');
    writer.FrameRate = 60;
    open(writer);
end

if splitFrontBack
    titles = ["Front thigh", "Back thigh"];
    frontMaskGlobal = (thetaAll <= (pi/2)) | (thetaAll >= (3*pi/2)); % centered near 0/2pi
    backMaskGlobal  = ~frontMaskGlobal; % near pi
    thetaWrapped = mod(thetaAll + pi, 2*pi) - pi; % wrap around pi -> [-pi, pi] with 0 at 0/2pi
    for ai = 1:2
        axSel = axs(ai);
        grid(axSel, 'on');
        xlabel(axSel, 'theta (rad)'); ylabel(axSel, 'axial (m)');
        if axSel == axFront
            xlim(axSel, [-pi/2, pi/2]);
            xticks(axSel, [-pi/2 0 pi/2]);
            xticklabels(axSel, {'-\pi/2','0','\pi/2'});
        else
            xlim(axSel, [pi/2, 3*pi/2]);
            xticks(axSel, [pi/2 pi 3*pi/2]);
            xticklabels(axSel, {'\pi/2','\pi','3\pi/2'});
        end
        ylim(axSel, [axLo, axHi]);
        set(axSel, 'XLimMode', 'manual', 'YLimMode', 'manual');
        axis(axSel, 'manual');
        title(axSel, titles(ai));
        caxis(axSel, [0 magMax]);
    end
    cb = colorbar(axs(end)); ylabel(cb, '|force| (N)');

    mask0 = time == tUnique(1);
    sizes0 = sMin + (magSm / magMax) * (sMax - sMin);
    scFront = scatter(axFront, thetaWrapped(mask0 & frontMaskGlobal), axialAll(mask0 & frontMaskGlobal), sizes0(mask0 & frontMaskGlobal), magSm(mask0 & frontMaskGlobal), 'filled', 'MarkerEdgeColor', 'k');
    scBack  = scatter(axBack,  thetaAll(mask0 & backMaskGlobal),  axialAll(mask0 & backMaskGlobal),  sizes0(mask0 & backMaskGlobal),  magSm(mask0 & backMaskGlobal),  'filled', 'MarkerEdgeColor', 'k');
    vidImg = [];
    if hasVideo && ~isempty(vidFrame)
        axes(axVideo); %#ok<LAXES>
        vidImg = imshow(vidFrame, 'Parent', axVideo);
        axis(axVideo, 'off');
        title(axVideo, 'Video');
    end

    for k = 1:nFrames
        mask = time == tUnique(k);
        sizes = sMin + (magSm(mask) / magMax) * (sMax - sMin);
        frontNow = mask & frontMaskGlobal;
        backNow  = mask & backMaskGlobal;

        set(scFront, 'XData', thetaWrapped(frontNow), ...
                     'YData', axialAll(frontNow), ...
                     'CData', magSm(frontNow), ...
                     'SizeData', sizes(frontMaskGlobal(mask)));
        set(scBack,  'XData', thetaAll(backNow),  ...
                     'YData', axialAll(backNow),  ...
                     'CData', magSm(backNow),  ...
                     'SizeData', sizes(backMaskGlobal(mask)));
        for axSel = axs
            if axSel == axFront
                set(axSel, 'XLim', [-pi/2, pi/2]);
            else
                set(axSel, 'XLim', [pi/2, 3*pi/2]);
            end
            set(axSel, 'YLim', [axLo, axHi]);
        end
        title(axFront, titles(1));
        title(axBack,  titles(2));
        if hasVideo && ~isempty(vidFrame)
            targetFrame = max(1, min(vidTotalFrames, round(tUnique(k) * vidFPS) + 1));
            while vidFrameIdx < targetFrame && hasFrame(vidObj)
                vidFrame = readFrame(vidObj);
                vidFrameIdx = vidFrameIdx + 1;
            end
            set(vidImg, 'CData', vidFrame);
            title(axVideo, 'Video');
        end
        drawnow;
        if ~isempty(writer)
            writeVideo(writer, getframe(fig));
        end
    end
else
    mask0 = time == tUnique(1);
    sizes0 = sMin + (magSm(mask0) / magMax) * (sMax - sMin);
    sc = scatter(ax, thetaAll(mask0), axialAll(mask0), sizes0, magSm(mask0), 'filled', 'MarkerEdgeColor', 'k');
    vidImg = [];
    if hasVideo && ~isempty(vidFrame)
        axes(axVideo); %#ok<LAXES>
        vidImg = imshow(vidFrame, 'Parent', axVideo);
        axis(axVideo, 'off');
        title(axVideo, 'Video');
    end

    for k = 1:nFrames
        mask = time == tUnique(k);
        sizes = sMin + (magSm(mask) / magMax) * (sMax - sMin);
        set(sc, 'XData', thetaAll(mask), ...
                'YData', axialAll(mask), ...
                'CData', magSm(mask), ...
                'SizeData', sizes);
        set(ax, 'XLim', [thetaLo, thetaHi], 'YLim', [axLo, axHi]);
        title(ax, '');
        if hasVideo && ~isempty(vidFrame)
            targetFrame = max(1, min(vidTotalFrames, round(tUnique(k) * vidFPS) + 1));
            while vidFrameIdx < targetFrame && hasFrame(vidObj)
                vidFrame = readFrame(vidObj);
                vidFrameIdx = vidFrameIdx + 1;
            end
            set(vidImg, 'CData', vidFrame);
            title(axVideo, 'Video');
        end
        drawnow;
        if ~isempty(writer)
            writeVideo(writer, getframe(fig));
        end
    end
end

if ~isempty(writer)
    close(writer);
    fprintf('Saved %s\n', outputVideo);
end
end
