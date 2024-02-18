import React from 'react';
import styles from './PageTitle.module.css';

const PageTitle: React.FC = () => {
    return (
      <header className={styles.titleSection}>
        <h1 className={styles.mainTitle}>University of California, Berkeley!</h1>
        <h2 className={styles.subtitle}>Professional Certificate in Machine Learning and Artificial Intelligence</h2>
        <h2 className={styles.subtitle}>Capstone Project!</h2>
        <h2 className={styles.subtitle}>Predictive Maintenance Modeling for Industrial Pump Using Machine Learning</h2>
        <p className={styles.subSubtitle}>
          "How can Machine Learning techniques be applied to predict imminent pump failures, 
          thereby minimizing downtime and extending the lifespan of industrial equipment?"
        </p>
        <p className={styles.author}>Author: Harri J Salomaa</p>
        <p className={styles.dataSource}>
          Data source: 
          <a href="https://www.kaggle.com/datasets/nphantawee/pump-sensor-data" target="_blank" rel="noopener noreferrer">
            Kaggle Pump Sensor Data
          </a>
        </p>
      </header>
    );
  };

  export default PageTitle;