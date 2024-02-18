import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine } from 'recharts';
import { format } from 'date-fns';
import styles from './SensorTrend.module.css';

interface DataPoint {
  time: string;
  [key: string]: string | number;
}

interface SensorTrendProps {
  data: DataPoint[];
  sensorName: string;
}

const StatusTrend: React.FC<SensorTrendProps> = ({ data, sensorName }) => {
    const dateFormatter = (date: string) => format(new Date(date), 'HH:mm');
    // console.log('StatusTrend', sensorName, data);
    
    // Render the chart only if data is available
    const renderChart = () => (
      <div className={styles.sensorTrendChart}>
        <LineChart width={800} height={200} data={data} key={new Date().getTime()} 
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444444"/>
            <XAxis dataKey="time" tickFormatter={dateFormatter} stroke="#cccccc"/>
            <YAxis type="number" domain={[0, 1]} allowDecimals={false}  stroke="#cccccc"/>
            <Tooltip
                contentStyle={{
                backgroundColor: "#333333", // Dark background for the tooltip
                borderColor: "#777777", // Dark border for the tooltip
                }}
                itemStyle={{ color: "#cccccc" }} // Light text color for tooltip items
            />
            <Legend
                wrapperStyle={{
                color: "#cccccc", // Light text color for legend
                }}
            />
            <Line type="monotone" dataKey={sensorName} stroke="#82ca9d" activeDot={{ r: 8 }} />
        </LineChart>
      </div>
    );
    
    return data && data.length > 0 ? renderChart() : null;
};

export default StatusTrend;
