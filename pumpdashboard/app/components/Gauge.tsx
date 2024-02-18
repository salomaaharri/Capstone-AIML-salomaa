import React, { useEffect, useState } from 'react';

interface Zone {
  low: number;
  high: number;
  color: string;
}

interface GaugeProps {
  size: number,
  background?: string,
  oldValue: number;
  newValue: number;
  min: number;
  max: number;
  step: number,
  zones: Zone[];
  label?: string; 
}

const Gauge: React.FC<GaugeProps> = ({ oldValue, newValue, min, max, step = 10, size = 100, background = "blue", zones, label }) => {
  const startAngle = -225; // Starting angle at -225 degrees
  const endAngle = 45; // Ending angle at 45 degrees
  const angleRange = endAngle - startAngle;
  const valueRange = max - min;
  // const relativeValue = value - min;
  // const angle = (relativeValue / valueRange) * angleRange + startAngle;
  const numTicks = (max - min) / step + 1

  // SVG styles and calculations
  const strokeWidth = 10;
  const radius = size / 2 - strokeWidth / 2;
  const tickLength = 5;
  const textOffsetPercentage = 0.21; // Adjust this percentage as needed
  const textOffset = -(radius * textOffsetPercentage);
  const fontSize = size * 0.06; 

  // const relativeValue = value - min;
  // const angle = (relativeValue / valueRange) * angleRange + startAngle;

  // State to manage the animated value
  const [animatedValue, setAnimatedValue] = useState(oldValue);
  // Calculate angle based on animatedValue instead of value
  const relativeAnimatedValue = animatedValue - min;
  const angle = (relativeAnimatedValue / valueRange) * angleRange + startAngle;
 

  useEffect(() => {
    // Trigger the animation when newValue changes
    const timer = setTimeout(() => {
      setAnimatedValue(newValue);
    }, 100); // Start the animation after a short delay

    return () => clearTimeout(timer);
  }, [newValue]);

  // Function to generate path for the arc segments of the gauge
  const getArcPath = (startValue: number, endValue: number, color: string, index: number) => {
    const valueToAngle = (value: number) => {
      return (value / valueRange) * angleRange + startAngle;
    };

    const startValueAngle = valueToAngle(startValue);
    const endValueAngle = valueToAngle(endValue);
    const arcAngle = Math.abs(endValueAngle - startValueAngle);

    // const largeArcFlag = endValue - startValue <= (max - min) / 2 ? "0" : "1";
    const largeArcFlag = arcAngle <= 180 ? "0" : "1";

    const startX = size / 2 + radius * Math.cos(Math.PI / 180 * startValueAngle);
    const startY = size / 2 + radius * Math.sin(Math.PI / 180 * startValueAngle);
    const endX = size / 2 + radius * Math.cos(Math.PI / 180 * endValueAngle);
    const endY = size / 2 + radius * Math.sin(Math.PI / 180 * endValueAngle);

    return (
      <path
        key={`zone-${color}-${index}`} // Unique key for each path
        d={`M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}`}
        stroke={color}
        fill="none"
        strokeWidth={strokeWidth}
      />
    );
  };

  // Function to generate tick marks
  // Function to generate tick marks and values
  const renderTicks = () => {
    const elements = [];
    const tickAngleIncrement = angleRange / (numTicks - 1);
    const tickRadius = radius - 10

    for (let i = 0; i < numTicks; i++) {
      const tickValue = min + (valueRange / (numTicks - 1)) * i;
      const tickAngle = i * tickAngleIncrement + startAngle;
      const x1 = size / 2 + tickRadius * Math.cos((tickAngle * Math.PI) / 180);
      const y1 = size / 2 + tickRadius * Math.sin((tickAngle * Math.PI) / 180);
      const x2 = size / 2 + (tickRadius + tickLength) * Math.cos((tickAngle * Math.PI) / 180);
      const y2 = size / 2 + (tickRadius + tickLength) * Math.sin((tickAngle * Math.PI) / 180);
      
      const textX = size / 2 + (tickRadius + tickLength + textOffset) * Math.cos((tickAngle * Math.PI) / 180);
      const textY = size / 2 + (tickRadius + tickLength + textOffset) * Math.sin((tickAngle * Math.PI) / 180);

      elements.push(
        <line
          key={`tick-${i}`}
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="white"
          strokeWidth="2"
        />
      );

      elements.push(
        <text
          key={`text-${i}`}
          x={textX}
          y={textY}
          fill="white"
          textAnchor="middle"
          alignmentBaseline="middle"
          fontSize={`${fontSize}px`}
        >
          {tickValue.toFixed(0)}
        </text>
      );
    }

    return elements;
  };

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ border: '2px solid #ffffff', transition: 'transform 1s ease-out' }}>

      {background !== 'none' && (
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          stroke="#00bfff"
          fill="none"
          strokeDasharray={radius * Math.PI * 2}
          transform={`rotate(${startAngle} ${size / 2} ${size / 2})`}
        />
      )}
      {zones.map((zone, index) => getArcPath(zone.low, zone.high, zone.color, index))}
      {renderTicks()}
      <line
        x1={size / 2}
        y1={size / 2}
        x2={size / 2 + radius * Math.cos(angle * Math.PI / 180)}
        y2={size / 2 + radius * Math.sin(angle * Math.PI / 180)}
        stroke="white"
        strokeWidth="2"
      />
      {/* Display current value */}
      <text
        x={size / 2}
        y={size / 2 - textOffset}
        fill="white"
        fontSize={`${fontSize}px`}
        textAnchor="middle"
        alignmentBaseline="central"
      >
        {`${newValue.toFixed(2)}`}
      </text>
      <text
        x={size / 2}
        y={size / 2 - textOffset + fontSize + 5}
        fill="white"
        fontSize={`${fontSize}px`}
        textAnchor="middle"
        alignmentBaseline="central"
      >
        {`${label ? ` ${label}` : ''}`}
      </text>
    </svg>
  );
};

export default Gauge;