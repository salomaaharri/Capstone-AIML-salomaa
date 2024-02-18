import React, { useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { format } from 'date-fns';
import styles from './SensorTrend.module.css';

// Define the structure of each data point
interface DataPoint {
  time: string;
  [key: string]: string | number;
}

// Define the props for SensorTrend
interface SensorTrendProps {
  data: DataPoint[];
  sensorName: string;
}

const SensorTrend: React.FC<SensorTrendProps> = ({ data, sensorName }) => {
  const dateFormatter = (date: string) => {
    // This will format the date to something like "2023-12-21"
    // You can adjust the format string as per your needs
    return format(new Date(date), 'HH:mm');
  };
  useEffect(() => {
    // Logic that should run when data updates
    // console.log(`Data for sensor ${sensorName} updated:`, data);
    // You can implement additional logic here if needed

  }, [data]);  // Dependency array, will re-run the effect when 'data' changes

  // console.log('StatusTrend', sensorName, data)

   return (
    <div className={styles.sensorTrendChart}> {/* Using CSS Module */}
      <LineChart width={800} height={200} data={data} key={new Date().getTime()} 
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" tickFormatter={dateFormatter} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey={sensorName} stroke="#8884d8" activeDot={{ r: 8 }} />
      </LineChart>
    </div>
  );
};

export default SensorTrend;