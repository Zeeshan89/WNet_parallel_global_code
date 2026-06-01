import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import numpy as np
from pathlib import Path
import calendar
import glob

class ETCTrackPlotter:
    """
    A class for plotting Extra-Tropical Cyclone (ETC) tracks from monthly data files.
    """
    
    def __init__(self, base_path, output_dir, min_track_length=5):
        """
        Initialize the ETC Track Plotter.
        
        Parameters:
        -----------
        base_path : str
            Base directory containing the ETC track files
        output_dir : str
            Directory where plots will be saved
        min_track_length : int
            Minimum number of points required for a valid track
        """
        self.base_path = base_path
        self.output_dir = output_dir
        self.min_track_length = min_track_length
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("Plot bounds: Auto-calculate from data")
    
    def parse_etc_file(self, file_path):
        """
        Parse an ETC tracking text file in our generated format and extract track information.
        
        Parameters:
        -----------
        file_path : str
            Path to the ETC tracking file
            
        Returns:
        --------
        list : List of track dictionaries
        """
        tracks = []
        current_track = None
        current_track_id = None
        
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return tracks
        
        for line in lines:
            line = line.strip()
            
            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse data line: TRACK_ID TIME LONGITUDE LATITUDE INTENSITY
            parts = line.split()
            if len(parts) >= 5:
                try:
                    track_id = int(parts[0])
                    time_str = parts[1]
                    lon = float(parts[2])
                    lat = float(parts[3])
                    intensity = float(parts[4])
                    
                    # Parse timestamp
                    timestamp = datetime.strptime(time_str, '%Y-%m-%d_%H:%M:%S')
                    
                    # If this is a new track, save the previous one and start a new one
                    if track_id != current_track_id:
                        # Save previous track if it exists
                        if current_track is not None and len(current_track['points']) > 0:
                            tracks.append(current_track)
                        
                        # Start new track
                        current_track_id = track_id
                        current_track = {
                            'track_id': track_id,
                            'points': [],
                            'start_time': timestamp,
                            'end_time': timestamp,
                            'duration': 0.0,
                            'total_distance': 0.0,
                            'max_intensity': intensity,
                            'min_intensity': intensity,
                            'num_detections': 0,
                            'is_strong': False
                        }
                    
                    # Add point to current track
                    current_track['points'].append({
                        'timestamp': timestamp,
                        'latitude': lat,
                        'longitude': lon,
                        'intensity': intensity
                    })
                    
                    # Update track metadata
                    current_track['end_time'] = timestamp
                    current_track['num_detections'] += 1
                    current_track['max_intensity'] = max(current_track['max_intensity'], intensity)
                    current_track['min_intensity'] = min(current_track['min_intensity'], intensity)
                    
                    # Calculate duration
                    duration_seconds = (current_track['end_time'] - current_track['start_time']).total_seconds()
                    current_track['duration'] = duration_seconds / 3600.0
                    
                    # Determine if strong
                    if intensity > 100:
                        current_track['is_strong'] = True
                        
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
        
        # Don't forget to add the last track
        if current_track is not None and len(current_track['points']) > 0:
            tracks.append(current_track)
        
        # Calculate total distance for each track
        for track in tracks:
            track['total_distance'] = self.calculate_track_distance(track['points'])
        
        return tracks
    
    def calculate_track_distance(self, points):
        """
        Calculate the total distance traveled along a track.
        
        Parameters:
        -----------
        points : list
            List of track points with lat/lon coordinates
            
        Returns:
        --------
        float : Total distance in kilometers
        """
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        earth_radius = 6371.0
        
        for i in range(1, len(points)):
            lat1 = np.radians(points[i-1]['latitude'])
            lon1 = np.radians(points[i-1]['longitude'])
            lat2 = np.radians(points[i]['latitude'])
            lon2 = np.radians(points[i]['longitude'])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            distance = earth_radius * c
            
            total_distance += distance
        
        return total_distance
    
    def filter_tracks(self, tracks, min_points=None):
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
        if min_points is None:
            min_points = self.min_track_length
            
        filtered_tracks = []
        
        for track in tracks:
            points = track['points']
            if len(points) >= min_points:
                filtered_tracks.append(track)
        
        return filtered_tracks
 
    def load_yearly_tracks(self, year):
        """
        Load tracks for a given year from yearly generated format.
        
        Parameters:
        -----------
        year : int
            Year to process
            
        Returns:
        --------
        list : List of tracks for the year
        """
        possible_patterns = [
            f"ETC_tracks_{year}_meteorological.txt",
            f"ETC_tracks_{year}.txt"
        ]
        
        for pattern in possible_patterns:
            file_path = os.path.join(self.base_path, pattern)
            if os.path.exists(file_path):
                print(f"Loading year {year}: {pattern}")
                try:
                    tracks = self.parse_etc_file(file_path)
                    filtered_tracks = self.filter_tracks(tracks)
                    print(f"  Found {len(filtered_tracks)} tracks for {year} (filtered from {len(tracks)} total)")
                    return filtered_tracks
                except Exception as e:
                    print(f"  Error loading {year}: {e}")
                    return []
        
        # Try to find with glob pattern
        pattern = os.path.join(self.base_path, f"*{year}*ETC*.txt")
        matching_files = glob.glob(pattern)
        if matching_files:
            file_path = matching_files[0]
            print(f"Loading year {year}: {os.path.basename(file_path)}")
            try:
                tracks = self.parse_etc_file(file_path)
                filtered_tracks = self.filter_tracks(tracks)
                print(f"  Found {len(filtered_tracks)} tracks for {year} (filtered from {len(tracks)} total)")
                return filtered_tracks
            except Exception as e:
                print(f"  Error loading {year}: {e}")
                return []
        else:
            print(f"No file found for year {year}")
            return []
    
    def calculate_year_extent(self, yearly_tracks):
        """
        Calculate the overall extent for all tracks in the year.
        Auto-calculates from data.
        
        Parameters:
        -----------
        yearly_tracks : list
            List of track data for the year
            
        Returns:
        --------
        list : [lon_min, lon_max, lat_min, lat_max]
        """
        all_lats = []
        all_lons = []
        
        for track in yearly_tracks:
            points = track['points']
            lats = [p['latitude'] for p in points]
            lons = [p['longitude'] for p in points]
            all_lats.extend(lats)
            all_lons.extend(lons)
        
        if not all_lats or not all_lons:
            return [-180, 180, -90, 90]
        
        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)
        
        # Add padding
        lat_padding = max((lat_max - lat_min) * 0.1, 5.0)
        lon_padding = max((lon_max - lon_min) * 0.1, 10.0)
        
        return [
            max(lon_min - lon_padding, -180), 
            min(lon_max + lon_padding, 180),
            max(lat_min - lat_padding, -90), 
            min(lat_max + lat_padding, 90)
        ]
    
    def split_track_at_dateline(self, lons, lats, threshold=170):
        """
        Split a track into segments when it crosses the dateline.
        
        Parameters:
        -----------
        lons : list
            Longitudes of track points
        lats : list
            Latitudes of track points
        threshold : float
            Longitude difference threshold to detect dateline crossing
        
        Returns:
        --------
        list : List of (segment_lons, segment_lats) tuples
        """
        segments = []
        current_seg_lons = [lons[0]]
        current_seg_lats = [lats[0]]
        
        for i in range(1, len(lons)):
            lon_diff = abs(lons[i] - lons[i-1])
            
            if lon_diff > threshold:
                if len(current_seg_lons) > 1:
                    segments.append((current_seg_lons, current_seg_lats))
                
                current_seg_lons = [lons[i]]
                current_seg_lats = [lats[i]]
            else:
                current_seg_lons.append(lons[i])
                current_seg_lats.append(lats[i])
        
        if len(current_seg_lons) > 0:
            segments.append((current_seg_lons, current_seg_lats))
        
        return segments
    
    def plot_annual_etc_tracks(self, yearly_tracks, year):
        """
        Create a single annual plot showing all ETC tracks from the entire year.
        Handles dateline crossing properly.
        
        Parameters:
        -----------
        yearly_tracks : list
            List of tracks for the year
        year : int
            Year of the data
        """
        # Calculate extent
        extent = self.calculate_year_extent(yearly_tracks)
        
        # Choose appropriate projection
        lon_range = extent[1] - extent[0]
        
        if lon_range > 180 or (extent[0] < -90 and extent[1] > 90):
            projection = ccrs.Robinson(central_longitude=0)
            use_global_projection = True
        else:
            projection = ccrs.PlateCarree()
            use_global_projection = False
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        ax = plt.subplot(1, 1, 1, projection=projection)
        
        # Set extent
        if not use_global_projection:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_global()
        
        # Add map features
        ax.add_feature(cfeature.LAND, alpha=0.7, color='lightgray')
        ax.add_feature(cfeature.OCEAN, alpha=0.3, color='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.7)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=not use_global_projection, linewidth=0.6, alpha=0.7)
        if not use_global_projection:
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
        
        # Track statistics
        annual_stats = {
            'total_tracks': 0,
            'strong_tracks': 0
        }
        
        # Plot all tracks
        for track in yearly_tracks:
            points = track['points']
            lats = [p['latitude'] for p in points]
            lons = [p['longitude'] for p in points]
            
            # Split track at dateline crossings
            track_segments = self.split_track_at_dateline(lons, lats)
            
            # Plot each segment
            for seg_lons, seg_lats in track_segments:
                ax.plot(
                    seg_lons, seg_lats,
                    color='red',
                    linewidth=0.8,
                    alpha=0.7,
                    transform=ccrs.PlateCarree()
                )
            
            # Mark start point
            ax.plot(
                lons[0], lats[0],
                marker='*',
                color='black',
                markersize=4,
                markeredgecolor='black',
                markeredgewidth=0.5,
                transform=ccrs.PlateCarree()
            )
            
            if track.get('is_strong', False):
                annual_stats['strong_tracks'] += 1
            
            annual_stats['total_tracks'] += 1
        
        # Add title
        title = f'Annual Extra-Tropical Cyclone (ETC) Tracks - {year}\n'
        title += f'Total: {annual_stats["total_tracks"]} tracks (≥{self.min_track_length} points) | '
        title += f'Strong: {annual_stats["strong_tracks"]}\n'
        title += f'Legend: Black stars (*): track start points | Red lines: track paths'
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, f'ETC_tracks_{year}_annual_min{self.min_track_length}pts_overall.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Annual plot saved: {output_path}")
        print(f"Annual {year} statistics:")
        print(f"  Total tracks: {annual_stats['total_tracks']}")
        print(f"  Strong tracks: {annual_stats['strong_tracks']}")
        
        return annual_stats
    
    def process_year(self, year):
        """
        Process ETC data for a specific year - creates annual plot.
        
        Parameters:
        -----------
        year : int
            Year to process
            
        Returns:
        --------
        dict : Dictionary containing processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing year {year} for ETC plots...")
        print(f"Using minimum track length: {self.min_track_length} points")
        print(f"{'='*60}")
        
        try:
            # Load yearly track data
            yearly_tracks = self.load_yearly_tracks(year)
            
            total_tracks = len(yearly_tracks)
            
            if total_tracks == 0:
                print(f"No tracks found for year {year}")
                return {'success': False, 'error': 'No tracks found'}
            
            print(f"Total tracks for year: {total_tracks}")
            
            # Create overall plot
            print("\n" + "="*40)
            print("Creating overall annual figure...")
            print("="*40)
            annual_stats = self.plot_annual_etc_tracks(yearly_tracks, year)
            
            return {
                'success': True,
                'year': year,
                'total_tracks': annual_stats['total_tracks'],
                'strong_tracks': annual_stats['strong_tracks'],
                'annual_stats': annual_stats
            }
            
        except Exception as e:
            print(f"Error processing year {year}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """
    Main function to run the ETC track plotting for multiple years.
    """
    # Set paths
    base_path = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial58/ETCs_enchanced_test_etc/ETC_track_files_yearly_5/"
    output_dir = "/lustre/orion/cli138/proj-shared/z7a/Storms_data/Storms_precipitation_notebooks/Percentile_ARs_detection/ARs_Hurricanes_ETCs_MCS/MCS_enhanced_trial58/ETCs_enchanced_test_etc/ETC_track_files_yearly_5/yearly_plots"
    
    # Year range configuration
    start_year = 1980
    end_year = 1990
    
    # Track filtering configuration
    min_track_length = 8
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print(f"ETC Track Annual Plotting Configuration:")
    print(f"- Base path: {base_path}")
    print(f"- Output directory: {output_dir}")
    print(f"- Year range: {start_year} to {end_year}")
    print(f"- Minimum track length: {min_track_length} points")
    print("- Plot bounds: Auto-calculate from data")
    
    # Create ETC plotter instance
    etc_plotter = ETCTrackPlotter(base_path, output_dir, min_track_length=min_track_length)
    
    # Process all years in range
    successful_years = 0
    failed_years = []
    total_tracks_all_years = 0
    
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*60}")
        print(f"Processing year {year}...")
        print(f"{'='*60}")
        
        result = etc_plotter.process_year(year)
        
        if result['success']:
            print(f"✓ Successfully processed year {year}")
            print(f"  Total tracks: {result['total_tracks']}")
            print(f"  Strong tracks: {result['strong_tracks']}")
            successful_years += 1
            total_tracks_all_years += result['total_tracks']
        else:
            print(f"✗ Failed to process year {year}: {result.get('error', 'Unknown error')}")
            failed_years.append(year)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Years processed: {start_year} to {end_year}")
    print(f"Successfully processed: {successful_years}/{end_year - start_year + 1} years")
    print(f"Total tracks across all years: {total_tracks_all_years}")
    
    if failed_years:
        print(f"Failed years: {failed_years}")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()