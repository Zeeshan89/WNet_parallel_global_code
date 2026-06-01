import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

def parse_hurricane_file(file_path):
    """
    Parse a hurricane tracking text file produced by HurricaneTracker.save_tracking_results
    and return a list of track dicts, each containing:
        duration    : float (hours)
        start_time  : datetime
        end_time    : datetime
        points      : list of point dicts
    """
    tracks = []
    try:
        with open(file_path, 'r') as fh:
            content = fh.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return tracks

    # ------------------------------------------------------------------ #
    # Split into per-track sections on the TRACK header line              #
    # ------------------------------------------------------------------ #
    section_pattern = re.compile(r'TRACK\s+\d+\s+\(Track\s+\d+\s+of\s+\d+\)')
    splits = section_pattern.split(content)
    track_sections = splits[1:]  # splits[0] is the file header

    print(f"Found {len(track_sections)} track sections in {os.path.basename(file_path)}")

    for sec_idx, section in enumerate(track_sections):
        # ---- scalar metadata ----------------------------------------- #
        duration_m   = re.search(r'Duration:\s*([\d.]+)\s*hours', section)
        start_time_m = re.search(r'Start Time:\s*(.+)',           section)
        end_time_m   = re.search(r'End Time:\s*(.+)',             section)

        if not (duration_m and start_time_m and end_time_m):
            # print(f"  Section {sec_idx+1}: missing metadata, skipping")
            continue

        def _parse_ts(raw):
            raw = raw.strip()
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
                try:
                    return datetime.strptime(raw, fmt)
                except ValueError:
                    pass
            raw_trimmed = re.sub(r'\.\d+$', '', raw)
            return datetime.strptime(raw_trimmed, '%Y-%m-%d %H:%M:%S')

        try:
            duration   = float(duration_m.group(1))
            start_time = _parse_ts(start_time_m.group(1))
            end_time   = _parse_ts(end_time_m.group(1))
        except Exception as e:
            # print(f"  Section {sec_idx+1}: metadata parse error ({e}), skipping")
            continue

        # ---- track points -------------------------------------------- #
        points = []
        in_data_block = False
        header_seen   = False

        for line in section.splitlines():
            stripped = line.strip()

            if 'Timestamp' in stripped and 'Latitude' in stripped:
                header_seen = True
                continue

            if header_seen and stripped.startswith('---'):
                in_data_block = True
                continue

            if in_data_block and (stripped.startswith('====') or
                                   stripped.startswith('TRACK')):
                break

            if not in_data_block or not stripped:
                continue

            try:
                ts_str  = line[0:20].strip()
                lat_str = line[20:32].strip()
                lon_str = line[32:44].strip()
                vor_str = line[44:56].strip()
                msl_str = line[56:68].strip()
                wnd_str = line[68:80].strip()

                def _safe_float(s):
                    try:
                        return float(s)
                    except (ValueError, TypeError):
                        return None

                timestamp = _parse_ts(ts_str)
                lat = _safe_float(lat_str)
                lon = _safe_float(lon_str)
                if lat is None or lon is None:
                    continue

                points.append({
                    'timestamp' : timestamp,
                    'latitude'  : lat,
                    'longitude' : lon,
                    'vorticity' : _safe_float(vor_str),
                    'mslp'      : _safe_float(msl_str),
                    'wind_speed': _safe_float(wnd_str),
                })
            except Exception:
                continue

        # print(f"  Section {sec_idx+1}: {len(points)} points, "
        #       f"{duration:.0f} h, {start_time.strftime('%Y-%m-%d')} → "
        #       f"{end_time.strftime('%Y-%m-%d')}")

        if points:
            tracks.append({
                'duration'  : duration,
                'start_time': start_time,
                'end_time'  : end_time,
                'points'    : points,
            })

    return tracks


def split_track_at_dateline(lons, lats):
    """
    Split track into segments when it crosses the dateline (180°/-180°).
    Returns list of (lon_segment, lat_segment) tuples.
    """
    segments = []
    current_lons = []
    current_lats = []

    for i in range(len(lons)):
        if i == 0:
            current_lons.append(lons[i])
            current_lats.append(lats[i])
            continue

        if abs(lons[i] - lons[i-1]) > 180:  # Dateline crossing
            if current_lons:
                segments.append((current_lons.copy(), current_lats.copy()))
            current_lons = [lons[i]]
            current_lats = [lats[i]]
        else:
            current_lons.append(lons[i])
            current_lats.append(lats[i])

    if current_lons:
        segments.append((current_lons, current_lats))

    return segments


def compute_path_km(points):
    """Haversine total path length in km for a list of point dicts."""
    total = 0.0
    for i in range(1, len(points)):
        lat1 = np.radians(points[i-1]['latitude'])
        lon1 = np.radians(points[i-1]['longitude'])
        lat2 = np.radians(points[i]['latitude'])
        lon2 = np.radians(points[i]['longitude'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        total += 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return total


def plot_hurricane_tracks(tracks, year, output_dir):
    """
    Create a spatial plot of all hurricane tracks for a given year.
    """
    extent = [-180.0, 180.0, -45.0, 45.0]

    colors = plt.cm.tab20.colors  # 20 distinct colors

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,      alpha=0.7, color='lightgray',  zorder=0)
    ax.add_feature(cfeature.OCEAN,     alpha=0.3, color='lightblue',  zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8,                  zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
    ax.add_feature(cfeature.STATES,    linewidth=0.3, alpha=0.5,      zorder=1)

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    if tracks:

        for i, track in enumerate(tracks):
            track_color = colors[i % len(colors)]
            points = track['points']
            lats = [p['latitude']  for p in points]
            lons = [p['longitude'] for p in points]

            segments = split_track_at_dateline(lons, lats)

            for seg_lons, seg_lats in segments:
                ax.plot(seg_lons, seg_lats,
                        color=track_color, linewidth=1, alpha=1,
                        transform=ccrs.PlateCarree(), zorder=2)

            if segments:
                first_seg_lons, first_seg_lats = segments[0]
                ax.plot(first_seg_lons[0], first_seg_lats[0],
                        marker='*', color=track_color, linewidth=1, markersize=10,
                        markeredgecolor='black', markeredgewidth=0.8,
                        transform=ccrs.PlateCarree(), zorder=3)

        total_distance = sum(track.get('total_path_km', compute_path_km(track['points']))
                             for track in tracks)
        avg_duration   = sum(track['duration'] for track in tracks) / len(tracks)

        plt.title(f'Hurricane Tracks - {year}\n'
                  f'Region: 180°W–180°E, 45°S–45°N | Tracks: {len(tracks)} | '
                  f'Avg Duration: {avg_duration:.1f}h | Total Distance: {total_distance:.0f}km',
                  fontsize=14, fontweight='bold', pad=20)
    else:
        plt.title(f'Hurricane Tracks - {year}\n'
                  f'Region: 180°W–180°E, 45°S–45°N | No Tracks Found',
                  fontsize=14, fontweight='bold', pad=20)

    output_path = os.path.join(output_dir, f'hurricane_tracks_{year}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Plot saved: {output_path}")

def render_hurricane_figure(tracks, year):
    """Build and return the matplotlib figure without saving."""
    extent = [-180.0, 180.0, -45.0, 45.0]
    colors = plt.cm.tab20.colors

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND,      alpha=0.7, color='lightgray',  zorder=0)
    ax.add_feature(cfeature.OCEAN,     alpha=0.3, color='lightblue',  zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8,                  zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
    ax.add_feature(cfeature.STATES,    linewidth=0.3, alpha=0.5,      zorder=1)

    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    if tracks:
        for i, track in enumerate(tracks):
            track_color = colors[i % len(colors)]
            points = track['points']
            lats = [p['latitude']  for p in points]
            lons = [p['longitude'] for p in points]

            segments = split_track_at_dateline(lons, lats)
            for seg_lons, seg_lats in segments:
                ax.plot(seg_lons, seg_lats,
                        color=track_color, linewidth=1, alpha=1,
                        transform=ccrs.PlateCarree(), zorder=2)

            if segments:
                first_seg_lons, first_seg_lats = segments[0]
                ax.plot(first_seg_lons[0], first_seg_lats[0],
                        marker='*', color=track_color, markersize=10,
                        markeredgecolor='black', markeredgewidth=0.8,
                        transform=ccrs.PlateCarree(), zorder=3)

        total_distance = sum(track.get('total_path_km', compute_path_km(track['points']))
                             for track in tracks)
        avg_duration = sum(track['duration'] for track in tracks) / len(tracks)
        plt.title(f'Hurricane Tracks - {year}\n'
                  f'Region: 180°W–180°E, 45°S–45°N | Tracks: {len(tracks)} | '
                  f'Avg Duration: {avg_duration:.1f}h | Total Distance: {total_distance:.0f}km',
                  fontsize=14, fontweight='bold', pad=20)
    else:
        plt.title(f'Hurricane Tracks - {year}\n'
                  f'Region: 180°W–180°E, 45°S–45°N | No Tracks Found',
                  fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def plot_hurricane_tracks(tracks, year, output_dir):
    """Render figure and save as PNG."""
    fig = render_hurricane_figure(tracks, year)
    output_path = os.path.join(output_dir, f'hurricane_tracks_{year}.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def save_all_years_pdf(output_dir, years_processed):
    """
    Combine all per-year hurricane track plots into a single high-quality PDF.
    Re-renders each plot as vector directly into PDF pages.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(output_dir, 'hurricane_tracks_all_years.pdf')
    print(f"\nGenerating high-quality PDF for {len(years_processed)} years...")

    with PdfPages(pdf_path) as pdf:
        for year in sorted(years_processed):
            folder_name = f"CONUS_year_{year}"
            file_name   = f"CONUS_hurricane_tracks_detailed_{year}.txt"
            file_path   = os.path.join(os.path.dirname(output_dir), folder_name, file_name)

            if not os.path.exists(file_path):
                # fallback: try to find the file
                folder_path = os.path.join(os.path.dirname(output_dir), folder_name)
                if os.path.exists(folder_path):
                    candidates = [f for f in os.listdir(folder_path)
                                  if str(year) in f and f.endswith('.txt')]
                    if candidates:
                        file_path = os.path.join(folder_path, candidates[0])
                    else:
                        print(f"  Skipping {year}: no data file found")
                        continue
                else:
                    print(f"  Skipping {year}: folder not found")
                    continue

            tracks = parse_hurricane_file(file_path)
            if not tracks:
                print(f"  Skipping {year}: no tracks parsed")
                continue

            for track in tracks:
                track['total_path_km'] = compute_path_km(track['points'])

            # Re-render the figure and save directly into PDF (vector quality)
            fig = render_hurricane_figure(tracks, year)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"  Added {year} to PDF")

    print(f"High-quality PDF saved: {pdf_path}")

def process_all_years(base_path, output_dir):
    """
    Process all years from 1980 to 2024, plotting ALL parsed tracks without filtering.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_stats = []

    print(f"Checking base directory: {base_path}")
    if os.path.exists(base_path):
        items = os.listdir(base_path)
        print(f"Items in base directory: {items[:10]}...")
        year_folders = [item for item in items if item.startswith('CONUS_year_')]
        print(f"Found {len(year_folders)} year folders")
        if year_folders:
            print(f"Sample year folders: {year_folders[:5]}")
    else:
        print(f"Base directory does not exist: {base_path}")
        return

    for year in range(1980, 2025):
        print(f"\n{'='*60}")
        print(f"Processing year {year}...")
        print(f"{'='*60}")

        folder_name = f"CONUS_year_{year}"
        file_name   = f"CONUS_hurricane_tracks_detailed_{year}.txt"
        file_path   = os.path.join(base_path, folder_name, file_name)
        folder_path = os.path.join(base_path, folder_name)

        print(f"Looking for: {file_path}")

        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            possible_folders = [item for item in os.listdir(base_path)
                                if str(year) in item and 'CONUS' in item]
            if possible_folders:
                print(f"Alternative folders found: {possible_folders}")
            continue

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            folder_files  = os.listdir(folder_path)
            possible_files = [f for f in folder_files if str(year) in f and f.endswith('.txt')]
            if possible_files:
                file_path = os.path.join(folder_path, possible_files[0])
                print(f"Using alternative file: {file_path}")
            else:
                continue

        try:
            print(f"File size: {os.path.getsize(file_path)} bytes")

            # Parse ALL tracks — no filtering applied
            tracks = parse_hurricane_file(file_path)
            print(f"Plotting all {len(tracks)} tracks (no filtering)")

            if tracks:
                # Compute path length for summary stats
                for track in tracks:
                    track['total_path_km'] = compute_path_km(track['points'])

                plot_hurricane_tracks(tracks, year, output_dir)

                summary_stats.append({
                    'year'              : year,
                    'total_tracks'      : len(tracks),
                    'total_path_km'     : sum(t['total_path_km'] for t in tracks),
                    'avg_duration_hours': sum(t['duration'] for t in tracks) / len(tracks),
                    'max_duration_hours': max(t['duration'] for t in tracks),
                    'avg_points_per_track': sum(len(t['points']) for t in tracks) / len(tracks),
                })
            else:
                print("No tracks found — check file format")

        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if summary_stats:
        df_summary   = pd.DataFrame(summary_stats)
        # summary_path = os.path.join(output_dir, 'hurricane_tracks_summary.csv')
        # df_summary.to_csv(summary_path, index=False)
        # print(f"\nSummary statistics saved: {summary_path}")

        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Years processed:            {len(summary_stats)}")
        print(f"Total tracks found:         {df_summary['total_tracks'].sum()}")
        print(f"Avg duration hours per year:{df_summary['avg_duration_hours'].mean():.1f}")
        print(f"Avg points per track:       {df_summary['avg_points_per_track'].mean():.1f}")
        save_all_years_pdf(output_dir, df_summary['year'].tolist())
    else:
        print("\nNo data was successfully processed. Please check file paths and formats.")



def main():
    base_path  = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/ETCs_enchanced_test_etc_21/Hurricanes_track_files_final_3"
    output_dir = "/lustre/orion/proj-shared/cli138/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial61/ETCs_enchanced_test_etc_21/Hurricanes_track_files_final_3/Plotting_results"

    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return

    print("Hurricane Track Spatial Plotting Script (no filtering)")
    print("="*50)
    print(f"Base path:        {base_path}")
    print(f"Output directory: {output_dir}")
    print("="*50)

    process_all_years(base_path, output_dir)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()