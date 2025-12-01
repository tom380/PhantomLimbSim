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
    posPlot = reconstruct_local(S, posWorld);
    coordLabel = 'local body frame';
    d = vecnorm(posPlot - posWorld, 2, 2);
    fprintf('Local vs world offset: mean %.4f, min %.4f, max %.4f (m)\n', mean(d), min(d), max(d));
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

function pos_local = reconstruct_local(S, pos_world)
% Prefer saved pos_local; otherwise rebuild from saved world pose.
n = size(pos_world, 1);
pos_local = pos_world;

if isfield(S, 'pos_local') && ~isempty(S.pos_local)
    pos_local = reshape(double(S.pos_local), [], 3);
    return;
end

if isfield(S, 'body_pos_world') && isfield(S, 'body_rot_world') ...
        && ~isempty(S.body_pos_world) && ~isempty(S.body_rot_world)

    body_pos = reshape(double(S.body_pos_world), [], 3);
    if size(body_pos, 1) == n
        rot = double(S.body_rot_world);
        pos_local = nan(n, 3);

        if ndims(rot) == 3 && size(rot, 1) == 3 && size(rot, 2) == 3 && size(rot, 3) == n
            for k = 1:n
                R = rot(:, :, k); % body->world
                delta = pos_world(k, :) - body_pos(k, :);
                pos_local(k, :) = (R.' * delta.').';
            end
        elseif ndims(rot) == 3 && size(rot, 2) == 3 && size(rot, 3) == 3 && size(rot, 1) == n
            for k = 1:n
                R = squeeze(rot(k, :, :));
                delta = pos_world(k, :) - body_pos(k, :);
                pos_local(k, :) = (R.' * delta.').';
            end
        else
            rot_raw = reshape(rot, [], 9);
            if size(rot_raw, 1) == n
                for k = 1:n
                    R = reshape(rot_raw(k, :), 3, 3);
                    delta = pos_world(k, :) - body_pos(k, :);
                    pos_local(k, :) = (R.' * delta.').';
                end
            else
                warning('body_rot_world length mismatch; using world coordinates');
                pos_local = pos_world;
            end
        end
    else
        warning('body_pos_world length mismatch; using world coordinates');
    end
end

if ~any(isfinite(pos_local), 'all')
    pos_local = pos_world;
end
end
