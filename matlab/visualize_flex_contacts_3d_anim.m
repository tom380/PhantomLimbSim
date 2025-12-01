function visualize_flex_contacts_3d_anim(matFile, outputVideo, maxFrames, targetFlex)
%VISUALIZE_FLEX_CONTACTS_3D_ANIM Animate flex contacts unwrapped to 2D.
%   visualize_flex_contacts_3d_anim()                     % uses flex_contacts_simple.mat
%   visualize_flex_contacts_3d_anim('file.mat')           % live playback only
%   visualize_flex_contacts_3d_anim('file.mat','out.mp4') % also saves MP4
%   visualize_flex_contacts_3d_anim('file.mat','',120)    % limit to 120 frames for faster playback
%   visualize_flex_contacts_3d_anim('file.mat','',120,3)  % only plot flex_id == 3 (e.g., thigh or shank)
%
% Uses PCA-based unwrap of local positions. No saved theta/axial required.

if nargin < 1 || isempty(matFile),   matFile   = 'flex_contacts_simple.mat'; end
if nargin < 2,                       outputVideo = []; end
if nargin < 3 || isempty(maxFrames), maxFrames = 240; end
if nargin < 4,                       targetFlex = []; end

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

% Sort by time
[time, order] = sort(time);
pos        = pos(order, :);
pos_local  = pos_local(order, :);
mag        = mag(order, :);
fid        = fid(order);

% Unwrap positions to cylindrical coordinates using PCA-derived axis (local frame)
center = mean(pos_local, 1);
rel = pos_local - center;
covMat = cov(rel);
[V, D] = eig(covMat);
[~, idxMax] = max(diag(D));
axisVec = V(:, idxMax);
axisVec = axisVec / norm(axisVec);

axialAll = rel * axisVec;
radial = rel - axialAll * axisVec.';
radialNorm = vecnorm(radial, 2, 2);
valid = radialNorm > 1e-12;
if ~any(valid)
    error('Cannot unwrap: radial extent is zero.');
end
[~, idxRef] = max(radialNorm);
refVec = radial(idxRef, :) / radialNorm(idxRef);
binormal = cross(axisVec, refVec);
binormal = binormal / norm(binormal);

thetaAll = zeros(size(axialAll));
thetaAll(valid) = atan2(radial(valid, :) * binormal.', radial(valid, :) * refVec.');
thetaAll = mod(thetaAll, 2*pi);

% Downsample unique times for playback speed
tUnique = unique(time);
if numel(tUnique) > maxFrames
    keepIdx = round(linspace(1, numel(tUnique), maxFrames));
    tUnique = tUnique(keepIdx);
end
nFrames = numel(tUnique);

fprintf('Animating %d frames from %s\n', nFrames, matFile);

% Fixed axes
thetaLo = 0; thetaHi = 2*pi;
axMin = min(axialAll);
axMax = max(axialAll);
axRange = axMax - axMin;
if ~isfinite(axRange) || axRange <= 0
    axRange = 1e-3;
end
pad = 0.05 * axRange;
axLo = axMin - pad;
axHi = axMax + pad;

fig = figure('Name', 'Flex contact animation');
ax  = axes('Parent', fig);
grid(ax, 'on');
xlabel(ax, 'theta (rad)'); ylabel(ax, 'axial (m)');
title(ax, 't = 0.000 s');
cb = colorbar(ax); ylabel(cb, '|force| (N)');
xlim(ax, [thetaLo, thetaHi]);
ylim(ax, [axLo, axHi]);
set(ax, 'XLimMode', 'manual', 'YLimMode', 'manual');
axis(ax, 'manual');

% first frame data
mask = time == tUnique(1);
sc = scatter(ax, thetaAll(mask), axialAll(mask), 20, mag(mask), 'filled');

writer = [];
if ~isempty(outputVideo)
    writer = VideoWriter(outputVideo, 'MPEG-4');
    writer.FrameRate = 60;
    open(writer);
end

for k = 1:nFrames
    mask = time == tUnique(k);
    set(sc, 'XData', thetaAll(mask), ...
            'YData', axialAll(mask), ...
            'CData', mag(mask));
    set(ax, 'XLim', [thetaLo, thetaHi], 'YLim', [axLo, axHi]);
    title(ax, sprintf('t = %.3f s', tUnique(k)));
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
