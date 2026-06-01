import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import numpy as np
from pathlib import Path

class ETCTrackPlotter:
    """
    A class for plotting Extra-Tropical Cyclone (ETC) tracks from annual data files.
    """
    
    def __init__(self, base_path, output_dir):
        """
        Initialize the ETC Track Plotter.
        
        Parameters:
        -----------
        base_path : str
            Base directory containing the ETC track files
        output_dir : str
            Directory where plots will be saved
        """
        self.base_path = base_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def parse_etc_file(self, file_path):
        """
        Parse an ETC tracking text file and extract track information.
        
        Parameters:
        -----------
        file_path : str
            Path to the ETC tracking file
            
        Returns:
        --------
        list : List of track dictionaries
        """
        tracks = []
        
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return tracks
        
        print(f"File content length: {len(content)} characters")
        
        # Debug: Print first few lines
        lines = content.split('\n')[:20]
        print("First 20 lines of file:")
        for i, line in enumerate(lines):
            print(f"{i+1:2d}: {line}")
        
        # Split by track sections - Updated patterns for annual files
        track_sections = re.split(r'TRACK \d+ \(Track \d+ of \d+\)', content)[1:]
        print(f"Found {len(track_sections)} track sections")
        
        # Alternative pattern if first doesn't work
        if len(track_sections) == 0:
            track_sections = re.split(r'TRACK \d+', content)[1:]
            print(f"Found {len(track_sections)} track sections using pattern 2")
        
        for i, section in enumerate(track_sections):
            print(f"\nProcessing track section {i+1}")
            print(f"Section length: {len(section)} characters")
            
            # Extract basic track info
            duration_match = re.search(r'Duration: ([\d.]+) hours', section)
            start_time_match = re.search(r'Start Time: (.+)', section)
            end_time_match = re.search(r'End Time: (.+)', section)
            num_detections_match = re.search(r'Number of detections: (\d+)', section)
            distance_match = re.search(r'Total distance traveled: ([\d.]+) km', section)
            
            # Check for special ETC types - Updated patterns
            is_arctic = ('Arctic ETC' in section or 
                        '** HIGH LATITUDE / ARCTIC ETC **' in section)
            is_strong = '** STRONG ETC SYSTEM **' in section
            
            print(f"Duration match: {duration_match.group(1) if duration_match else 'None'}")
            print(f"Start time match: {start_time_match.group(1) if start_time_match else 'None'}")
            print(f"End time match: {end_time_match.group(1) if end_time_match else 'None'}")
            print(f"Arctic ETC: {is_arctic}")
            print(f"Strong ETC: {is_strong}")
            
            if not all([duration_match, start_time_match, end_time_match]):
                print("Missing basic track info, skipping section")
                continue
                
            try:
                duration = float(duration_match.group(1))
                start_time = datetime.strptime(start_time_match.group(1), '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(end_time_match.group(1), '%Y-%m-%d %H:%M:%S')
                num_detections = int(num_detections_match.group(1)) if num_detections_match else None
                total_distance = float(distance_match.group(1)) if distance_match else None
            except Exception as e:
                print(f"Error parsing basic track info: {e}")
                continue
            
            # Extract track points
            track_points = []
            lines = section.split('\n')
            
            # Find the start of data rows (after the detailed track points header)
            data_start = False
            header_found = False
            
            for line_num, line in enumerate(lines):
                # Look for the detailed track points header
                if 'Timestamp' in line and 'Latitude' in line and 'Longitude' in line:
                    header_found = True
                    continue
                
                if '-----' in line and header_found:
                    data_start = True
                    continue
                
                if data_start and line.strip():
                    # Check if we've reached the end of this track
                    if line.strip().startswith('TRACK') or line.strip().startswith('===='):
                        break
                        
                    # Parse data line - ETC format has more columns than hurricane format
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            # Extract timestamp, latitude, longitude
                            timestamp_str = f"{parts[0]} {parts[1]}"
                            lat = float(parts[2])
                            lon = float(parts[3])
                            
                            # Additional ETC-specific parameters
                            vorticity = float(parts[4]) if len(parts) > 4 and parts[4] != 'N/A' else None
                            mslp = float(parts[5]) if len(parts) > 5 and parts[5] != 'N/A' else None
                            wind_speed = float(parts[6]) if len(parts) > 6 and parts[6] != 'N/A' else None
                            area = float(parts[7]) if len(parts) > 7 and parts[7] != 'N/A' else None
                            max_intensity = float(parts[8]) if len(parts) > 8 and parts[8] != 'N/A' else None
                            mean_intensity = float(parts[9]) if len(parts) > 9 and parts[9] != 'N/A' else None
                            
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                            
                            track_points.append({
                                'timestamp': timestamp,
                                'latitude': lat,
                                'longitude': lon,
                                'vorticity': vorticity,
                                'mslp': mslp,
                                'wind_speed': wind_speed,
                                'area': area,
                                'max_intensity': max_intensity,
                                'mean_intensity': mean_intensity
                            })
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing line '{line}': {e}")
                            continue
            
            print(f"Found {len(track_points)} track points")
            
            if track_points:
                tracks.append({
                    'duration': duration,
                    'start_time': start_time,
                    'end_time': end_time,
                    'num_detections': num_detections,
                    'total_distance': total_distance,
                    'is_arctic': is_arctic,
                    'is_strong': is_strong,
                    'points': track_points
                })
        
        return tracks
    
    def filter_tracks(self, tracks, min_points=4):
        """
        Filter tracks based on minimum number of points.
        
        Parameters:
        -----------
        tracks : list
            List of track dictionaries
        min_points : int
            Minimum number of points required for a valid track
            
        Returns:
        --------
        list : Filtered list of tracks
        """
        filtered_tracks = []
        
        for track in tracks:
            points = track['points']
            if len(points) >= min_points:
                filtered_tracks.append(track)
        
        return filtered_tracks
    
    def plot_etc_tracks(self, tracks, year):
        if not tracks:
            print(f"No tracks to plot for year {year}")
            return

        # ── Fixed region: Lon 0 to -180 (Pacific/Americas), Lat 20 to 90 ──
        REGION_LON_MIN = -180
        REGION_LON_MAX = 180
        REGION_LAT_MIN = -90
        REGION_LAT_MAX = 90

        extent = [REGION_LON_MIN, REGION_LON_MAX, REGION_LAT_MIN, REGION_LAT_MAX]

        # Count track types (still loop through all tracks for stats)
        arctic_tracks = 0
        strong_tracks  = 0
        for track in tracks:
            if track.get('is_arctic', False): arctic_tracks += 1
            if track.get('is_strong', False): strong_tracks  += 1

        # ── Figure setup ──
        fig = plt.figure(figsize=(16, 12))
        ax  = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Map features
        ax.add_feature(cfeature.LAND,      alpha=0.7, color='lightgray')
        ax.add_feature(cfeature.OCEAN,     alpha=0.3, color='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.5)
        ax.add_feature(cfeature.STATES,    linewidth=0.3, alpha=0.5)

        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        regular_tracks = 0

        for track in tracks:
            points = track['points']
            lats = [p['latitude']  for p in points]
            lons = [p['longitude'] for p in points]

            # ── Skip tracks entirely outside the region ──
            if (max(lats) < REGION_LAT_MIN or min(lats) > REGION_LAT_MAX or
                max(lons) < REGION_LON_MIN or min(lons) > REGION_LON_MAX):
                continue

            ax.plot(lons, lats, color='red', linewidth=1.0, alpha=0.8,
                    transform=ccrs.PlateCarree())

            if track.get('is_arctic', False):
                ax.plot(lons[0], lats[0], marker='D', color='green',
                        markersize=4, markeredgecolor='black', markeredgewidth=0.8,
                        transform=ccrs.PlateCarree())
            else:
                regular_tracks += 1
                ax.plot(lons[0], lats[0], marker='*', color='black',
                        markersize=6, markeredgecolor='black', markeredgewidth=0.8,
                        transform=ccrs.PlateCarree())

        plt.title(
            f'Extra-Tropical Cyclone (ETC) Tracks - {year}\n'
            f'Total tracks: {len(tracks)} | Regular: {regular_tracks} | '
            f'Arctic: {arctic_tracks} | Strong: {strong_tracks}\n'
            f'Region: Lon 0°–180°W | Lat 20°–90°N  |  Regular (*) | Arctic (◆)',
            fontsize=14, fontweight='bold'
        )

        output_path = os.path.join(self.output_dir, f'ETC_tracks_{year}_annual.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_path}")
        
    def process_year(self, year):
        """
        Process ETC data for a specific year - Updated to match hurricane tracking structure.
        
        Parameters:
        -----------
        year : int
            Year to process
            
        Returns:
        --------
        dict : Dictionary containing processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing year {year}...")
        print(f"{'='*60}")
        
        # Construct paths similar to hurricane tracking - year-based directory structure
        year_dir = f"ETC_year_{year}"
        file_name = f"ETC_tracks_detailed_{year}.txt"
        
        # Full paths
        year_path = os.path.join(self.base_path, year_dir)
        file_path = os.path.join(year_path, file_name)
        
        print(f"Year directory: {year_dir}")
        print(f"Looking for file: {file_name}")
        print(f"Year path: {year_path}")
        print(f"Full file path: {file_path}")
        
        # Check if year directory exists
        if not os.path.exists(year_path):
            print(f"Year directory not found: {year_path}")
            
            # List available year directories for debugging
            if os.path.exists(self.base_path):
                available_dirs = [d for d in os.listdir(self.base_path) 
                                if os.path.isdir(os.path.join(self.base_path, d)) and 'year' in d.lower()]
                print(f"Available year directories: {available_dirs}")
            
            return {'success': False, 'error': f'Year directory not found: {year_dir}'}
        
        # Check if file exists in year directory
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            
            # List files in year directory for debugging
            if os.path.exists(year_path):
                available_files = os.listdir(year_path)
                print(f"Available files in {year_dir}: {available_files}")
                
                # Look for alternative file names
                possible_files = [f for f in available_files 
                                if str(year) in f and 'detailed' in f and f.endswith('.txt')]
                
                if possible_files:
                    print(f"Found alternative files: {possible_files}")
                    file_path = os.path.join(year_path, possible_files[0])
                    print(f"Using alternative file: {file_path}")
                else:
                    print(f"No suitable ETC files found in {year_dir}")
                    return {'success': False, 'error': f'No ETC file found for year {year}'}
            else:
                return {'success': False, 'error': f'Year directory not accessible: {year_dir}'}
        
        try:
            print(f"File exists: {os.path.exists(file_path)}")
            print(f"File size: {os.path.getsize(file_path)} bytes")
            
            # Parse tracks
            tracks = self.parse_etc_file(file_path)
            print(f"Found {len(tracks)} total tracks")
            
            if len(tracks) > 0:
                # Filter tracks
                filtered_tracks = self.filter_tracks(tracks, min_points=4)
                print(f"Plotting {len(filtered_tracks)} tracks (after filtering)")
                
                # Create plot
                self.plot_etc_tracks(filtered_tracks, year)
                
                return {
                    'success': True,
                    'year': year,
                    'total_tracks': len(tracks),
                    'plotted_tracks': len(filtered_tracks),
                    'arctic_tracks': sum(1 for track in filtered_tracks if track.get('is_arctic', False)),
                    'strong_tracks': sum(1 for track in filtered_tracks if track.get('is_strong', False))
                }
            else:
                print("No tracks found - check file format")
                return {'success': False, 'error': 'No tracks found in file'}
            
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def process_all_years(self, start_year=1980, end_year=2025):
        """
        Process all years in the specified range - Updated to match hurricane tracking structure.
        
        Parameters:
        -----------
        start_year : int
            Starting year (inclusive)
        end_year : int
            Ending year (exclusive)
        """
        print("ETC Track Annual Plotting Script")
        print("="*60)
        print(f"Base path: {self.base_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Processing years: {start_year} to {end_year-1}")
        print("="*60)
        
        all_results = []
        
        # First, check what's actually in the base directory
        print(f"\nChecking base directory: {self.base_path}")
        if os.path.exists(self.base_path):
            items = os.listdir(self.base_path)
            print(f"Items in base directory: {len(items)} files/directories")
            
            # Check for year directories
            year_dirs = [item for item in items 
                        if os.path.isdir(os.path.join(self.base_path, item)) and 'year' in item.lower()]
            print(f"Found {len(year_dirs)} year directories")
            if year_dirs:
                print(f"Sample year directories: {sorted(year_dirs)[:10]}")
            
            # Check for specific years in the requested range
            available_years = []
            for year in range(start_year, end_year):
                year_dir = f"ETC_year_{year}"
                year_path = os.path.join(self.base_path, year_dir)
                if os.path.exists(year_path):
                    available_years.append(year)
            
            print(f"Available years in range {start_year}-{end_year-1}: {len(available_years)} years")
            if available_years:
                print(f"Year range: {min(available_years)} to {max(available_years)}")
        else:
            print(f"Base directory does not exist: {self.base_path}")
            return
        
        # Process each year
        processed_years = 0
        failed_years = []
        
        for year in range(start_year, end_year):
            result = self.process_year(year)
            if result['success']:
                all_results.append(result)
                processed_years += 1
                print(f"✓ Successfully processed year {year}")
            else:
                failed_years.append((year, result.get('error', 'Unknown error')))
                print(f"✗ Failed to process year {year}: {result.get('error', 'Unknown error')}")
        
        # Create summary statistics
        if all_results:
            print(f"\Plots statistics saved:")

            if failed_years:
                print(f"\nFailed years:")

            
        else:
            print("\nNo data was successfully processed. Please check file paths and formats.")
            print(f"All {len(failed_years)} years failed processing:")
            for year, error in failed_years:
                print(f"  {year}: {error}")

def main():
    """
    Main function to run the ETC track plotting.
    """
    # Set paths - update these to match your directory structure
    base_path = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial58/ETCs_enchanced_test_etc/ETC_track_files_yearly"
    output_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial58/ETCs_enchanced_test_etc/ETC_track_files_yearly/ETC_plots"
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        print("Please update the base_path variable with the correct path to your ETC data.")
        return
    
    # Create ETC plotter instance
    etc_plotter = ETCTrackPlotter(base_path, output_dir)
    
    # Process all years (adjust range as needed)
    etc_plotter.process_all_years(start_year=1980, end_year=2024)  # Start with a small range for testing
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()