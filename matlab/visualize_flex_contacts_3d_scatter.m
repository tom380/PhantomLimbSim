function visualize_flex_contacts_3d_scatter(matFile, outputVideo, maxFrames, targetFlex, smoothWindowSec, maxContacts, sizeScale, splitFrontBack, syncVideo)
%VISUALIZE_FLEX_CONTACTS_3D_SCATTER Unwrapped scatter with size = |force|.
%
% Key performance fix:
% - Pre-bin contacts into frame index lists (cell array) so each frame does NOT scan all N.
%
% Other fixes:
% - O(N) smoothing grouping via accumarray (when enabled).
% - Robust cancellation (figure close / Ctrl+C) and VideoWriter cleanup.
% - Progress prints + optional waitbar.

if nargin < 1 || isempty(matFile),   matFile   = 'flex_contacts_simple.mat'; end
if nargin < 2,                       outputVideo = []; end
if nargin < 3 || isempty(maxFrames), maxFrames = 240; end
if nargin < 4,                       targetFlex = []; end
if nargin < 5 || isempty(smoothWindowSec), smoothWindowSec = 0.03; end
if nargin < 6 || isempty(maxContacts), maxContacts = 4e5; end
if nargin < 7 || isempty(sizeScale), sizeScale = 1.0; end
if nargin < 8 || isempty(splitFrontBack), splitFrontBack = false; end
if nargin < 9, syncVideo = []; end

tAll = tic;

fprintf('[1/11] Loading MAT: %s\n', matFile);
S = load(matFile);

fprintf('[2/11] Validating fields...\n');
for f = ["time","flex_id","pos","force_world","normal"]
    if ~isfield(S, f)
        error("Field '%s' missing in %s", f, matFile);
    end
end

fprintf('[3/11] Extracting arrays...\n');
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
    fprintf('[4/11] Filtering flex_id == %d...\n', targetFlex);
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
else
    fprintf('[4/11] No flex filter.\n');
end

% Optional contact cap
if maxContacts > 0 && numel(time) > maxContacts
    fprintf('[5/11] Downsampling contacts (cap=%g)...\n', maxContacts);
    stride = ceil(numel(time) / maxContacts);
    keepIdx = 1:stride:numel(time);
    time       = time(keepIdx);
    pos        = pos(keepIdx, :);
    pos_local  = pos_local(keepIdx, :);
    forceW     = forceW(keepIdx, :);
    mag        = mag(keepIdx, :);
    fid        = fid(keepIdx);
    fprintf('        Downsampled: stride %d (kept %d of %d)\n', stride, numel(time), numel(S.time));
else
    fprintf('[5/11] No downsampling (N=%d, cap=%g).\n', numel(time), maxContacts);
end

fprintf('[6/11] Sorting by time (N=%d)...\n', numel(time));
[time, order] = sort(time);
pos        = pos(order, :);
pos_local  = pos_local(order, :);
mag        = mag(order, :);
fid        = fid(order);

fprintf('[7/11] Estimating dt + unique times...\n');
tUniqueAll = unique(time);
if numel(tUniqueAll) >= 2
    dtEstimate = median(diff(tUniqueAll));
else
    d = diff(time);
    d = d(isfinite(d) & d > 0);
    if isempty(d), dtEstimate = 1/60; else, dtEstimate = median(d); end
end
if ~isfinite(dtEstimate) || dtEstimate <= 0
    dtEstimate = 1/60;
end
fprintf('        dtEstimate ~= %.6g sec, unique times = %d\n', dtEstimate, numel(tUniqueAll));

% Smooth |force|
magSm = mag;
if smoothWindowSec > 0
    fprintf('[8/11] Smoothing magnitudes (window=%.3g sec)...\n', smoothWindowSec);
    tSmooth = tic;
    winSamples = max(1, round(smoothWindowSec / dtEstimate));
    fprintf('        winSamples = %d\n', winSamples);

    if winSamples > 1
        posTol = 1e-5;
        posKey = round(pos_local / posTol) * posTol;

        [~, ~, posIdx] = unique(posKey, 'rows', 'stable');
        nGroups = max(posIdx);
        fprintf('        Groups = %d\n', nGroups);

        idxCells = accumarray(posIdx, (1:numel(posIdx)).', [], @(v){v});
        lastPrint = tic;
        stepPrint = max(1, round(nGroups * 0.05));

        for g = 1:nGroups
            idx = idxCells{g};
            if numel(idx) > 1
                magSm(idx) = movmean(mag(idx), winSamples, 'Endpoints', 'shrink');
            end
            if g == 1 || g == nGroups || mod(g, stepPrint) == 0 || toc(lastPrint) > 2
                fprintf('        Smoothing: %6.2f%% (%d/%d)\n', 100*g/nGroups, g, nGroups);
                lastPrint = tic;
            end
        end
    end
    fprintf('        Smoothing done in %.2f s\n', toc(tSmooth));
else
    fprintf('[8/11] Smoothing disabled (smoothWindowSec=0).\n');
end

fprintf('[9/11] Unwrapping to (theta, axial) per flex...\n');
tUnwrap = tic;
thetaAll = nan(size(time));
axialAll = nan(size(time));
flexVals = unique(fid(:)).';
axisVecKnown = [0; 0; 1];
refVecKnown  = [1; 0; 0];

for iFlex = 1:numel(flexVals)
    flexVal = flexVals(iFlex);
    maskFlex = fid == flexVal;
    if nnz(maskFlex) == 0, continue; end

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

    refVec = refVecKnown - axisVec.' * refVecKnown * axisVec;
    if norm(refVec) < 1e-12
        firstValid = find(valid, 1);
        refVec = radial(firstValid, :)' / radialNorm(firstValid);
    else
        refVec = refVec / norm(refVec);
    end

    binormal = cross(axisVec, refVec);
    if ~any(binormal), binormal = [0; 1; 0]; end
    binormal = binormal / norm(binormal);

    theta = zeros(size(axial));
    theta(valid) = atan2(radial(valid, :) * binormal, radial(valid, :) * refVec);
    theta = mod(theta, 2*pi);

    thetaAll(maskFlex) = theta;
    axialAll(maskFlex) = axial;

    fprintf('        Flex unwrap: %d/%d\n', iFlex, numel(flexVals));
end
fprintf('        Unwrap done in %.2f s\n', toc(tUnwrap));

% Downsample unique times for playback speed
tUnique = tUniqueAll;
if numel(tUnique) > maxFrames
    keepIdx = round(linspace(1, numel(tUnique), maxFrames));
    tUnique = tUnique(keepIdx);
end
nFrames = numel(tUnique);
fprintf('[10/11] Animating %d frames...\n', nFrames);

% -------- PERFORMANCE FIX: pre-bin indices per frame --------
fprintf('[11/11] Pre-binning contacts into frames...\n');
tBin = tic;

% Map each contact time to nearest selected frame time.
% Since time is sorted and tUnique is sorted, interp1 is efficient and avoids O(N*nFrames) scans.
frameOfContact = round(interp1(tUnique, 1:nFrames, time, 'nearest', 'extrap'));
frameOfContact = max(1, min(nFrames, frameOfContact));

frameIdx = accumarray(frameOfContact, (1:numel(time)).', [nFrames, 1], @(v){v}, {[]});

% Print distribution (helps diagnose huge frames)
counts = cellfun(@numel, frameIdx);
fprintf('        Binning done in %.2f s. Contacts/frame: min=%d, median=%d, max=%d\n', ...
    toc(tBin), min(counts), round(median(counts)), max(counts));

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
if ~isfinite(axRange) || axRange <= 0, axRange = 1e-3; end
pad = 0.05 * axRange;
axLo = axMin - pad;
axHi = axMax + pad;

% Figure / video setup
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
    fprintf('        Opening sync video: %s\n', syncVideo);
    vidObj = VideoReader(syncVideo);
    vidFPS = vidObj.FrameRate;
    vidTotalFrames = max(1, ceil(vidObj.Duration * vidFPS));
    if hasFrame(vidObj)
        vidFrame = readFrame(vidObj);
    end
end

% Layout
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
if magMax <= 0 || ~isfinite(magMax), magMax = 1; end
sMin = 8 * sizeScale;
sMax = 60 * sizeScale;

writer = [];
cleanupWriter = [];
if ~isempty(outputVideo)
    fprintf('        Opening VideoWriter: %s\n', outputVideo);
    writer = VideoWriter(outputVideo, 'MPEG-4');
    writer.FrameRate = 60;
    open(writer);
    cleanupWriter = onCleanup(@() safeCloseWriter(writer)); %#ok<NASGU>
end

% Progress UI
useWaitbar = usejava('desktop');
wb = [];
if useWaitbar
    wb = waitbar(0, 'Starting animation...', 'Name', 'visualize_flex_contacts_3d_scatter');
end
cleanupWB = onCleanup(@() safeCloseWaitbar(wb)); %#ok<NASGU>

tAnim = tic;
lastPrint = tic;
printEverySec = 2;
framePrintStep = max(1, round(nFrames * 0.05));

% Masks for front/back
if splitFrontBack
    titles = ["Front thigh", "Back thigh"];
    frontMaskGlobal = (thetaAll <= (pi/2)) | (thetaAll >= (3*pi/2));
    backMaskGlobal  = ~frontMaskGlobal;
    thetaWrapped = mod(thetaAll + pi, 2*pi) - pi;

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

    % init with first frame indices
    idx0 = frameIdx{1};
    if isempty(idx0), idx0 = 1; end

    front0 = idx0(frontMaskGlobal(idx0));
    back0  = idx0(backMaskGlobal(idx0));

    sizesFront0 = sMin + (magSm(front0) / magMax) * (sMax - sMin);
    sizesBack0  = sMin + (magSm(back0)  / magMax) * (sMax - sMin);

    scFront = scatter(axFront, thetaWrapped(front0), axialAll(front0), sizesFront0, magSm(front0), ...
        'filled', 'MarkerEdgeColor', 'k');
    scBack  = scatter(axBack,  thetaAll(back0),     axialAll(back0),  sizesBack0,  magSm(back0), ...
        'filled', 'MarkerEdgeColor', 'k');

    vidImg = [];
    if hasVideo && ~isempty(vidFrame)
        axes(axVideo); %#ok<LAXES>
        vidImg = imshow(vidFrame, 'Parent', axVideo);
        axis(axVideo, 'off');
        title(axVideo, 'Video');
    end

    try
        for k = 1:nFrames
            if ~isgraphics(fig)
                warning('Figure closed by user. Stopping animation.');
                break;
            end

            idx = frameIdx{k};
            if isempty(idx)
                % No contacts for this frame; still update progress/video
                idxFront = [];
                idxBack  = [];
            else
                idxFront = idx(frontMaskGlobal(idx));
                idxBack  = idx(backMaskGlobal(idx));
            end

            sizesFront = sMin + (magSm(idxFront) / magMax) * (sMax - sMin);
            sizesBack  = sMin + (magSm(idxBack)  / magMax) * (sMax - sMin);

            set(scFront, 'XData', thetaWrapped(idxFront), ...
                         'YData', axialAll(idxFront), ...
                         'CData', magSm(idxFront), ...
                         'SizeData', sizesFront);

            set(scBack,  'XData', thetaAll(idxBack), ...
                         'YData', axialAll(idxBack), ...
                         'CData', magSm(idxBack), ...
                         'SizeData', sizesBack);

            title(axFront, titles(1));
            title(axBack,  titles(2));

            if hasVideo && ~isempty(vidFrame)
                targetFrame = max(1, min(vidTotalFrames, round(tUnique(k) * vidFPS) + 1));
                while vidFrameIdx < targetFrame && hasFrame(vidObj)
                    vidFrame = readFrame(vidObj);
                    vidFrameIdx = vidFrameIdx + 1;
                end
                if ~isempty(vidImg) && ~isempty(vidFrame)
                    set(vidImg, 'CData', vidFrame);
                end
                title(axVideo, 'Video');
            end

            drawnow;

            if ~isempty(writer)
                if ~isgraphics(fig), break; end
                writeVideo(writer, getframe(fig));
            end

            % Progress
            if ~isempty(wb) && isgraphics(wb)
                waitbar(k/nFrames, wb, sprintf('Frame %d/%d (%.1f%%)', k, nFrames, 100*k/nFrames));
            end
            if k == 1 || k == nFrames || mod(k, framePrintStep) == 0 || toc(lastPrint) > printEverySec
                fprintf('        Frame %4d/%d (%.1f%%), elapsed %.1fs, contacts=%d\n', ...
                    k, nFrames, 100*k/nFrames, toc(tAnim), numel(idx));
                lastPrint = tic;
            end
        end
    catch ME
        if strcmp(ME.identifier, 'MATLAB:getframe:InvalidFigureOrAxes') || ...
           strcmp(ME.identifier, 'MATLAB:class:InvalidHandle') || ...
           strcmp(ME.identifier, 'MATLAB:graphics:Interrupted')
            warning('Animation interrupted. Exiting cleanly.');
        else
            rethrow(ME);
        end
    end

else
    % Single panel
    grid(ax, 'on');
    xlabel(ax, 'theta (rad)'); ylabel(ax, 'axial (m)');
    xlim(ax, [thetaLo, thetaHi]);
    ylim(ax, [axLo, axHi]);
    set(ax, 'XLimMode', 'manual', 'YLimMode', 'manual');
    axis(ax, 'manual');
    caxis(ax, [0 magMax]);
    cb = colorbar(ax); ylabel(cb, '|force| (N)');

    idx0 = frameIdx{1};
    if isempty(idx0), idx0 = 1; end
    sizes0 = sMin + (magSm(idx0) / magMax) * (sMax - sMin);
    sc = scatter(ax, thetaAll(idx0), axialAll(idx0), sizes0, magSm(idx0), 'filled', 'MarkerEdgeColor', 'k');

    vidImg = [];
    if hasVideo && ~isempty(vidFrame)
        axes(axVideo); %#ok<LAXES>
        vidImg = imshow(vidFrame, 'Parent', axVideo);
        axis(axVideo, 'off');
        title(axVideo, 'Video');
    end

    try
        for k = 1:nFrames
            if ~isgraphics(fig)
                warning('Figure closed by user. Stopping animation.');
                break;
            end

            idx = frameIdx{k};
            sizes = sMin + (magSm(idx) / magMax) * (sMax - sMin);

            set(sc, 'XData', thetaAll(idx), ...
                    'YData', axialAll(idx), ...
                    'CData', magSm(idx), ...
                    'SizeData', sizes);

            if hasVideo && ~isempty(vidFrame)
                targetFrame = max(1, min(vidTotalFrames, round(tUnique(k) * vidFPS) + 1));
                while vidFrameIdx < targetFrame && hasFrame(vidObj)
                    vidFrame = readFrame(vidObj);
                    vidFrameIdx = vidFrameIdx + 1;
                end
                if ~isempty(vidImg) && ~isempty(vidFrame)
                    set(vidImg, 'CData', vidFrame);
                end
                title(axVideo, 'Video');
            end

            drawnow;

            if ~isempty(writer)
                if ~isgraphics(fig), break; end
                writeVideo(writer, getframe(fig));
            end

            if ~isempty(wb) && isgraphics(wb)
                waitbar(k/nFrames, wb, sprintf('Frame %d/%d (%.1f%%)', k, nFrames, 100*k/nFrames));
            end
            if k == 1 || k == nFrames || mod(k, framePrintStep) == 0 || toc(lastPrint) > printEverySec
                fprintf('        Frame %4d/%d (%.1f%%), elapsed %.1fs, contacts=%d\n', ...
                    k, nFrames, 100*k/nFrames, toc(tAnim), numel(idx));
                lastPrint = tic;
            end
        end
    catch ME
        if strcmp(ME.identifier, 'MATLAB:getframe:InvalidFigureOrAxes') || ...
           strcmp(ME.identifier, 'MATLAB:class:InvalidHandle') || ...
           strcmp(ME.identifier, 'MATLAB:graphics:Interrupted')
            warning('Animation interrupted. Exiting cleanly.');
        else
            rethrow(ME);
        end
    end
end

fprintf('Done. Total elapsed: %.2fs\n', toc(tAll));
if ~isempty(outputVideo) && ~isempty(writer)
    fprintf('Output video target: %s (written frames depend on when run stopped)\n', outputVideo);
end

end


% ---------- Helpers ----------
function safeCloseWriter(w)
try
    if ~isempty(w)
        close(w);
    end
catch
end
end

function safeCloseWaitbar(wb)
try
    if ~isempty(wb) && isgraphics(wb)
        close(wb);
    end
catch
end
end
