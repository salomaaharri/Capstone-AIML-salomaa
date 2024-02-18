'use client'
import React, { useState, useEffect, useRef } from 'react';
import styles from './page.module.css'
import Gauge from './components/Gauge'
import PageTitle from './components/PageTitle'
import StatusIndicator from './components/StatusIndicator'
import SensorTrend from './components/SensorTrend';
import StatusTrend from './components/StatusTrend';

type MaxSensorValues = {
  [key: string]: number;
};

// Assuming each data point has a value and a timestamp
interface DataPoint {
  time: string; // Timestamp for the data point
  [sensorName: string]: string | number; // Sensor name as key and its value
}

// Historical data for each sensor
interface SensorDataHistory {
  [sensorName: string]: DataPoint[];
}

interface DangerZoneData {
  sensor: string;
  lower_threshold: number;
  upper_threshold: number;
  significance: string;
}

interface DangerZone {
  sensor: string;
  lowerThreshold: number;
  upperThreshold: number;
  significance: string;
}

export default function Home() {
  // const [data, setData] = useState<DataPoint[]>([]);
  const [machineStatus, setMachineStatus] = useState<'NORMAL' | 'BROKEN'>('NORMAL');
  const [predictedMachineStatus, setPredictedMachineStatus] = useState<'NORMAL' | 'BROKEN'>('NORMAL');
  const [sensorValues, setSensorValues] = useState([]);
  const [previousSensorValues, setPreviousSensorValues] = useState([]);
  const [sensorNames, setSensorNames] = useState([]);
  const [gaugeCount, setGaugeCount] = useState(0);
  const [gaugeSize, setGaugeSize] = useState(100);
  const [maxSensorValues, setMaxSensorValues] = useState<MaxSensorValues>({});
  const [sensorDataHistory, setSensorDataHistory] = useState<SensorDataHistory>({});
  const [dangerZones, setDangerZones] = useState<DangerZone[]>([]);
  const MAX_DATA_POINTS = 1000;

  const min = 0
  const size = 150

  function calculateStep(maxValue: number) {
    if (maxValue <= 10) {
      return 1;
    } else if (maxValue <= 20) {
      return 2;
    } else if (maxValue <= 50) {
      return 5;
    } else if (maxValue <= 100) {
      return 10;
    } else if (maxValue <= 200) {
      return 20;
    } else if (maxValue <= 500) {
      return 50;
    } else if (maxValue <= 1000) {
      return 100;
    } else if (maxValue <= 1500) {
      return 200;
    } else if (maxValue <= 2000) {
      return 200;
    } else {
      return 100;
    }
  }

  useEffect(() => {
    // Set the initial value once the component mounts
    const handleResize = () => {
        const height = window.innerHeight;
        setGaugeSize(height * 0.2); 
    };

    // Call handleResize initially and then set up the resize event listener
    handleResize();
    window.addEventListener('resize', handleResize);

    // Remove event listener on cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    // console.log("Updated sensor names:", sensorNames);
    // console.log("Updated sensor values:", sensorValues);
  }, [sensorNames, sensorValues]); // Only re-run the effect if sensorNames or sensorValues changes
  
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765");
  
    // console.log('useEffect connecting to websocket')

    ws.onopen = () => {
      // console.log("Connected to the server");
    };
  
    ws.onmessage = (event) => {
      // console.log("Raw data received:", event.data);
      try {
        const parsedData = JSON.parse(event.data);
        const timestamp = parsedData.timestamp; 
        const actualStatus = parsedData.machine_status === 'BROKEN' ? 1 : 0;
        const predictedStatus = parsedData.prediction === 'BROKEN' ? 1 : 0;
    
        // Extract sensor values and update the gauge values
        const newsensorValues = parsedData.sensor_data;
        // Use the max values received from the server
        if (parsedData.max_values) {
          setMaxSensorValues(currentMaxValues => {
            const updatedMaxValues: MaxSensorValues = { ...currentMaxValues };
            for (const [sensorName, maxValue] of Object.entries(parsedData.max_values)) {
              updatedMaxValues[sensorName] = maxValue as number;
            }
            return updatedMaxValues;
          });
        }

        // Save danger zones if they're included in the incoming message
        if (parsedData.danger_zones) {
          setDangerZones(parsedData.danger_zones.map((zone: DangerZoneData) => ({
            sensor: zone.sensor,
            lowerThreshold: zone.lower_threshold,
            upperThreshold: zone.upper_threshold,
            significance: zone.significance
          })));
        }

        // console.log(parsedData)

        if (parsedData.prediction) {
          setPredictedMachineStatus(parsedData.prediction)
        }

        if (parsedData.machine_status) {
          setMachineStatus(parsedData.machine_status)
        }

        if (parsedData.sensor_data) {

          setPreviousSensorValues(sensorValues);

          const newSensorNames = parsedData.sensor_data.map(([name, _]: [string, number]) => name);
          const newSensorValues = parsedData.sensor_data.map(([_, value]: [string, number]) => value);
      
          setSensorNames(newSensorNames);
          setSensorValues(newSensorValues);
          setGaugeCount(newSensorNames.length);

        }
        // Update historical data
        setSensorDataHistory(currentHistory => {
          const newHistory = { ...currentHistory };
          const timestamp = parsedData.timestamp; // Use the timestamp from the server
          parsedData.sensor_data.forEach(([name, value]: [string, number]) => {
            const dataPoint: DataPoint = { time: timestamp, [name]: value };
            if (!newHistory[name]) {
              newHistory[name] = [];
            }
            newHistory[name].push(dataPoint);
            if (newHistory[name].length > MAX_DATA_POINTS) {
              newHistory[name].shift(); // Remove the oldest data point
            }
          });
          // Update for machine status and predicted machine status
          ['machine_status', 'predicted_machine_status'].forEach((statusKey, index) => {
            const statusValue = index === 0 ? actualStatus : predictedStatus;
            // console.log(statusKey, statusValue)
            const dataPoint = { time: timestamp, [statusKey]: statusValue };
            if (!newHistory[statusKey]) {
              newHistory[statusKey] = [];
            }
            newHistory[statusKey].push(dataPoint);
            if (newHistory[statusKey].length > MAX_DATA_POINTS) {
              newHistory[statusKey].shift();
            }
          });
          return newHistory;
        });
        // Rest of your logic
      } catch (error) {
        console.error("Error parsing JSON:", error);
      }
    };
  
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
  
    ws.onclose = () => {
      console.log("Disconnected from the server");
    };
  
    return () => {
      ws.close();
    };
  }, []);

  return (
    <main className={styles.main}>
      <PageTitle />
      <div className={styles.sensorContainer}>
        <div className={styles.sensorGaugeContainer}>
          <div className={styles.sensorRow}>
              <StatusIndicator label='Status' value={machineStatus} size={gaugeSize}/>
          </div>
          {/* Machine Status Trend */}
          <div className={styles.machineStatusTrendContainer}>
            <StatusTrend
              data={sensorDataHistory['machine_status']}
              sensorName="machine_status"
            />
          </div>
        </div>
        <div className={styles.sensorGaugeContainer}>
          <div className={styles.sensorRow}>
              <StatusIndicator label='Prediction' value={predictedMachineStatus} size={gaugeSize}/>
          </div>
          <div className={styles.machineStatusTrendContainer}>
            <StatusTrend
              data={sensorDataHistory['predicted_machine_status']}
              sensorName="predicted_machine_status"
            />
          </div>
        </div>
        {sensorNames.map((sensorName, index) => {
          const oldValue = previousSensorValues[index] || 0; // Use previous value
          const newValue = sensorValues[index] || 0; // Use new value
          const maxValue = maxSensorValues[sensorName] || 100;
          const step = calculateStep(maxValue);
          // Find the danger zone for the current sensor
          const dangerZone: DangerZone | undefined = dangerZones.find(zone => zone.sensor === sensorName);

          // If there's no danger zone info for a sensor, default to some values
          // const lowDangerValue = dangerZone ? dangerZone.lowerThreshold : 0.7 * maxValue;
          // const highDangerValue = dangerZone ? dangerZone.upperThreshold : maxValue;
          // 6.2006523 51.986446 80
          let lowDangerValue: number;
          let highDangerValue: number;
          
          if (dangerZone) {
            // If dangerZone is defined, use its thresholds
            lowDangerValue = 0.7 * dangerZone.lowerThreshold; // Assuming you want to use 70% of lowerThreshold
            highDangerValue = dangerZone.upperThreshold;
          } else {
            // If dangerZone is undefined, fall back to some default values
            lowDangerValue = 0.7 * maxValue; // Some default proportional to maxValue
            highDangerValue = maxValue; // Default to maxValue
          }

          // console.log(lowDangerValue, highDangerValue, maxValue)

          return (
            <div className={styles.sensorGaugeContainer} key={sensorName}>
              <div className={styles.sensorRow}>
                <Gauge
                  oldValue={oldValue}
                  newValue={newValue}
                  min={0}
                  max={maxValue}
                  step={step}
                  size={gaugeSize}
                  background="none"
                  zones={[
                    { low: lowDangerValue, high: highDangerValue, color: 'red' } // Danger zone
                  ]}
                  label={sensorName}
                />
              </div>
              <div className={styles.sensorTrendContainer}>
                <SensorTrend
                  data={sensorDataHistory[sensorName]}
                  sensorName={sensorName}
                />
              </div>
            </div>
          );
        })}
      </div>
      <div className={styles.trendSection}>
        {/* Additional content here */}
      </div>
    </main>
  );  
};