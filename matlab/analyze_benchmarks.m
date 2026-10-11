%% Smart Vision Monitor — Benchmark Analysis in MATLAB
% Reads benchmark data exported by Python (reports/benchmarks.csv),
% plots latency vs. image resolution, calculates throughput scaling,
% evaluates bandwidth consumption, and validates numerical statistics.

clear; clc; close all;

csv_path = fullfile('..', 'reports', 'benchmarks.csv');
if ~isfile(csv_path)
    csv_path = 'reports/benchmarks.csv';
end

if ~isfile(csv_path)
    error('Benchmark file not found: %s. Run "python scripts/benchmark.py" first.', csv_path);
end

fprintf('Loading benchmark records from %s...\n', csv_path);
data = readtable(csv_path);
disp(data);

% ─── Extract Columns ───
img_sizes    = data.image_size;
mean_latency = data.mean_total_ms;
p50_infer    = data.p50_inference_ms;
p95_infer    = data.p95_inference_ms;
throughput   = data.throughput_fps;
peak_ram     = data.peak_ram_mb;

has_bandwidth = ismember('bandwidth_mbps', data.Properties.VariableNames);
if has_bandwidth
    bandwidth_mbps = data.bandwidth_mbps;
    frame_bytes    = data.frame_bytes;
else
    bandwidth_mbps = zeros(size(throughput));
    frame_bytes    = zeros(size(throughput));
end

% ─── Metric Calculations ───
if height(data) >= 2
    speedup_ratio = mean_latency(2) / mean_latency(1);
    pct_change = ((mean_latency(2) - mean_latency(1)) / mean_latency(1)) * 100;
    fprintf('\n--- Scaling Analysis ---\n');
    fprintf('Resolution %dx%d vs %dx%d:\n', img_sizes(1), img_sizes(1), img_sizes(2), img_sizes(2));
    fprintf('  Latency increase factor : %.2fx (%.1f%% increase)\n', speedup_ratio, pct_change);
    fprintf('  Throughput reduction   : %.1f FPS -> %.1f FPS\n', throughput(1), throughput(2));
    if has_bandwidth
        fprintf('  Bandwidth throughput   : %.2f MB/s vs %.2f MB/s\n', bandwidth_mbps(1), bandwidth_mbps(2));
    end
end

% ─── Visualization ───
figure('Name', 'Smart Vision Monitor - Performance Scaling', 'NumberTitle', 'off', ...
       'Position', [100, 150, 1200, 420]);

% Subplot 1: Latency by Image Size
subplot(1, 3, 1);
bar(img_sizes, [p50_infer, p95_infer, mean_latency]);
xlabel('Image Dimension (pixels)');
ylabel('Latency (ms)');
title('Latency Comparison (CPU)');
legend({'p50 Inference', 'p95 Inference', 'Mean Total'}, 'Location', 'northwest');
grid on;

% Subplot 2: Throughput FPS & RAM
subplot(1, 3, 2);
yyaxis left;
plot(img_sizes, throughput, '-o', 'LineWidth', 2, 'Color', [0.1, 0.6, 0.2]);
ylabel('Throughput (Frames Per Second)');
ylim([0, max(throughput) * 1.3]);

yyaxis right;
plot(img_sizes, peak_ram, '-s', 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2]);
ylabel('Peak Host RAM (MB)');
ylim([0, max(peak_ram) * 1.3]);

xlabel('Image Dimension (pixels)');
title('Throughput & Host Memory');
legend({'Throughput (FPS)', 'RAM (MB)'}, 'Location', 'northeast');
grid on;

% Subplot 3: Bandwidth (MB/s) & Frame Size
subplot(1, 3, 3);
if has_bandwidth
    yyaxis left;
    plot(img_sizes, bandwidth_mbps, '-^', 'LineWidth', 2, 'Color', [0.2, 0.4, 0.8]);
    ylabel('Bandwidth (MB/s)');
    ylim([0, max(bandwidth_mbps) * 1.3 + 0.1]);

    yyaxis right;
    plot(img_sizes, frame_bytes / 1024, '-d', 'LineWidth', 2, 'Color', [0.8, 0.5, 0.1]);
    ylabel('Raw Frame Size (KB)');
    ylim([0, max(frame_bytes / 1024) * 1.3 + 1]);

    xlabel('Image Dimension (pixels)');
    title('Bandwidth & Data Rate');
    legend({'Bandwidth (MB/s)', 'Frame Size (KB)'}, 'Location', 'northwest');
    grid on;
else
    text(0.5, 0.5, 'Bandwidth data not available', 'HorizontalAlignment', 'center');
    title('Bandwidth & Data Rate');
end

fprintf('\nPlots generated successfully.\n');
