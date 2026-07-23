import React from 'react';
import './globals.css';

export const metadata = {
  title: 'PredicChurn | AI Churn Prediction',
  description: 'Predict and prevent customer churn with advanced machine learning.',
};

export default function Home() {
  return (
    <>
      <div className="blob blob-1"></div>
      <div className="blob blob-2"></div>
      
      <header className="header glass">
        <div style={{ fontWeight: 'bold', fontSize: '1.25rem' }}>
          <span className="gradient-text">Predic</span>Churn
        </div>
        <nav style={{ display: 'flex', gap: '24px' }}>
          <a href="#dashboard" style={{ color: 'var(--foreground)', textDecoration: 'none', fontWeight: 500 }}>Dashboard</a>
          <a href="#models" style={{ color: 'var(--foreground)', textDecoration: 'none', fontWeight: 500 }}>Models</a>
          <a href="#reports" style={{ color: 'var(--foreground)', textDecoration: 'none', fontWeight: 500 }}>Reports</a>
        </nav>
        <button className="btn btn-primary">Connect Data</button>
      </header>

      <main className="main-content">
        <section className="hero">
          <h1>Stop Churn Before <br/><span className="gradient-text">It Happens</span></h1>
          <p>
            Our advanced machine learning platform predicts which customers are at risk of leaving,
            allowing you to take proactive measures and boost retention rates.
          </p>
          <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
            <button className="btn btn-primary">View Dashboard</button>
            <button className="btn glass">Run Analysis</button>
          </div>
        </section>

        <section id="dashboard" style={{ marginTop: '60px' }}>
          <h2 style={{ marginBottom: '32px', fontSize: '2rem' }}>Live Overview</h2>
          
          <div className="dashboard-grid">
            <div className="card glass">
              <div className="stat-label">Total Customers</div>
              <div className="stat-value">12,450</div>
              <div style={{ color: '#10b981', fontSize: '0.875rem', fontWeight: 600 }}>+5.2% this month</div>
            </div>
            
            <div className="card glass">
              <div className="stat-label">Predicted Churn Rate</div>
              <div className="stat-value gradient-text">8.4%</div>
              <div style={{ color: '#ef4444', fontSize: '0.875rem', fontWeight: 600 }}>+1.1% vs last month</div>
            </div>
            
            <div className="card glass">
              <div className="stat-label">At Risk Revenue</div>
              <div className="stat-value">$142,500</div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: '65%' }}></div>
              </div>
            </div>
          </div>
          
          <div style={{ marginTop: '24px', display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
             <div className="card glass" style={{ flex: '1 1 60%' }}>
               <h3 style={{ marginBottom: '24px', fontSize: '1.25rem' }}>High Risk Customers</h3>
               <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
                 <thead>
                   <tr style={{ borderBottom: '1px solid var(--border)', color: '#94a3b8' }}>
                     <th style={{ padding: '12px 0', fontWeight: 600 }}>Customer</th>
                     <th style={{ fontWeight: 600 }}>Plan</th>
                     <th style={{ fontWeight: 600 }}>Risk Score</th>
                     <th style={{ fontWeight: 600 }}>Action</th>
                   </tr>
                 </thead>
                 <tbody>
                   <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                     <td style={{ padding: '16px 0', fontWeight: '500' }}>Acme Corp</td>
                     <td style={{ color: '#cbd5e1' }}>Enterprise</td>
                     <td><span style={{ color: '#ef4444', fontWeight: 'bold' }}>92%</span></td>
                     <td><button className="btn glass" style={{ padding: '6px 12px', fontSize: '0.875rem' }}>Engage</button></td>
                   </tr>
                   <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                     <td style={{ padding: '16px 0', fontWeight: '500' }}>Global Tech</td>
                     <td style={{ color: '#cbd5e1' }}>Pro</td>
                     <td><span style={{ color: '#f59e0b', fontWeight: 'bold' }}>78%</span></td>
                     <td><button className="btn glass" style={{ padding: '6px 12px', fontSize: '0.875rem' }}>Engage</button></td>
                   </tr>
                   <tr>
                     <td style={{ padding: '16px 0', fontWeight: '500' }}>Startup Inc</td>
                     <td style={{ color: '#cbd5e1' }}>Basic</td>
                     <td><span style={{ color: '#f59e0b', fontWeight: 'bold' }}>71%</span></td>
                     <td><button className="btn glass" style={{ padding: '6px 12px', fontSize: '0.875rem' }}>Engage</button></td>
                   </tr>
                 </tbody>
               </table>
             </div>
             
             <div className="card glass" style={{ flex: '1 1 30%' }}>
               <h3 style={{ marginBottom: '24px', fontSize: '1.25rem' }}>Key Churn Factors</h3>
               <div style={{ marginBottom: '20px' }}>
                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontWeight: 500 }}>
                   <span>Low Usage Activity</span>
                   <span style={{ color: '#94a3b8' }}>42%</span>
                 </div>
                 <div className="progress-bar"><div className="progress-fill" style={{ width: '42%', background: 'linear-gradient(to right, #ef4444, #f43f5e)' }}></div></div>
               </div>
               <div style={{ marginBottom: '20px' }}>
                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontWeight: 500 }}>
                   <span>Support Tickets Open</span>
                   <span style={{ color: '#94a3b8' }}>28%</span>
                 </div>
                 <div className="progress-bar"><div className="progress-fill" style={{ width: '28%', background: 'linear-gradient(to right, #f59e0b, #fbbf24)' }}></div></div>
               </div>
               <div style={{ marginBottom: '20px' }}>
                 <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontWeight: 500 }}>
                   <span>Payment Failed</span>
                   <span style={{ color: '#94a3b8' }}>15%</span>
                 </div>
                 <div className="progress-bar"><div className="progress-fill" style={{ width: '15%', background: 'linear-gradient(to right, #3b82f6, #60a5fa)' }}></div></div>
               </div>
             </div>
          </div>
        </section>
      </main>
    </>
  );
}
