#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p Analysis

# Step 1: Merge all monthly files into one file
echo "Merging all monthly files..."
cdo mergetime era5dara_*_AR_detection.nc Analysis/AR_detection_merged.nc

# Step 2: Convert 6-hourly data to daily data (e.g., if AR=1, else 0)
echo "Converting 6-hourly data to daily data (AR day if any 6hr slot has AR)..."
cdo daymax Analysis/AR_detection_merged.nc Analysis/AR_detection_daily.nc

# Step 3: Count AR days per year (sum of AR days per year)
echo "Counting AR days per year..."
cdo yearsum Analysis/AR_detection_daily.nc Analysis/AR_detection_ARDaysPerYear.nc

# Step 4: Calculate average AR days per year
echo "Calculating average AR days per year..."
cdo timmean Analysis/AR_detection_ARDaysPerYear.nc Analysis/AR_detection_AvgARDaysPerYear.nc

echo "Processing complete! Results are in the Analysis directory."