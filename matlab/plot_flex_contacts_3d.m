function plot_flex_contacts_3d(matFile, targetFlex, useLocal, maxPoints)
%PLOT_FLEX_CONTACTS_3D Quick 3D scatter of contacts for one flex_id.
%   plot_flex_contacts_3d()                      % loads flex_contacts_simple.mat, flex_id=0, world coords
%   plot_flex_contacts_3d('file.mat',1,true)     % flex_id=1 (shank), uses pos_local
%   plot_flex_contacts_3d('file.mat',0,false,2e4)% limit to 20k points for speed

if nargin < 1 || isempty(matFile),  matFile = 'flex_contacts_simple.mat'; end
if nargin < 2,                      targetFlex = 0; end
if nargin < 3 || isempty(useLocal), useLocal = false; end
if nargin < 4 || isempty(maxPoints), maxPoints = 5e4; end

S = load(matFile);
for f = ["time","flex_id","pos","force_world"]
    if ~isfield(S, f)
        error("Field '%s' missing in %s", f, matFile);
    end
end

fid = S.flex_id(:);
mask = fid == targetFlex;
if ~any(mask)
    error('No contacts found for flex_id == %d', targetFlex);
end

posWorld = double(S.pos);
posPlot = posWorld;
coordLabel = 'world';
if useLocal
    if isfield(S, 'pos_local') && ~isempty(S.pos_local)
        posPlot = double(S.pos_local);
        coordLabel = 'local body frame';
        d = vecnorm(posPlot - posWorld, 2, 2);
        fprintf('Local vs world offset: mean %.4f, min %.4f, max %.4f (m)\n', mean(d), min(d), max(d));
    else
        warning('pos_local not found; using world coordinates');
    end
end

posPlot = posPlot(mask, :);
mag = vecnorm(double(S.force_world(mask, :)), 2, 2);
time = double(S.time(mask));

% Downsample if too many points
nPts = size(posPlot, 1);
if nPts > maxPoints
    idx = round(linspace(1, nPts, maxPoints));
    posPlot = posPlot(idx, :);
    mag = mag(idx);
    time = time(idx);
end

fig = figure('Name', sprintf('Flex %d contacts (%s)', targetFlex, coordLabel));
ax = axes('Parent', fig);
sc = scatter3(ax, posPlot(:,1), posPlot(:,2), posPlot(:,3), 10, mag, 'filled');
cb = colorbar(ax); ylabel(cb, '|force| (N)');
axis(ax, 'equal');
grid(ax, 'on');
xlabel(ax, 'x'); ylabel(ax, 'y'); zlabel(ax, 'z');
title(ax, sprintf('flex_id = %d (%s) | n = %d', targetFlex, coordLabel, size(posPlot,1)));

% Small helper to scrub through time if needed
disp('Tip: Use Data Cursor or rotate3d in the figure to inspect individual points.');
end
