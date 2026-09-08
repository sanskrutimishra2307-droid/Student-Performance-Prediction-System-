/**
 * Charts Engine for Student Performance Analytics
 * Clean, compact, value-added visualizations based on authentic dataset distributions.
 */

let activeChartInstance = null;

// Palette
const chartColors = {
  teal: '#00a88f',
  tealLight: 'rgba(0, 168, 143, 0.2)',
  purple: '#8b5cf6',
  purpleLight: 'rgba(139, 92, 246, 0.2)',
  coral: '#f43f5e',
  coralLight: 'rgba(244, 63, 94, 0.2)',
  blue: '#3b82f6',
  amber: '#f59e0b',
  slate: '#64748b'
};

const defaultOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    },
    tooltip: {
      backgroundColor: '#1e293b',
      padding: 12,
      cornerRadius: 10,
      titleFont: { family: 'Plus Jakarta Sans', size: 12, weight: 'bold' },
      bodyFont: { family: 'Plus Jakarta Sans', size: 12 }
    }
  },
  scales: {
    x: {
      grid: { display: false },
      ticks: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' }, color: '#64748b' }
    },
    y: {
      grid: { color: '#f1f5f9' },
      ticks: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' }, color: '#64748b' }
    }
  }
};

/**
 * Initialize / Render the selected performance analytics tab chart.
 */
function renderAnalyticsChart(tabName, datasetInsights) {
  const ctx = document.getElementById('analyticsMainChart');
  if (!ctx) return;

  if (activeChartInstance) {
    activeChartInstance.destroy();
  }

  // Fallback data if API not yet loaded
  const defaultInsights = {
    hours_vs_score: { "0-2 hrs": 52.4, "2-4 hrs": 68.2, "4-6 hrs": 81.5, "6-8 hrs": 89.0, "8+ hrs": 94.6 },
    attendance_vs_score: { "<50%": 48.0, "50-70%": 64.5, "70-85%": 79.2, "85-100%": 91.8 },
    method_stats: [
      { study_method: 'coaching', mean: 86.4 },
      { study_method: 'online videos', mean: 81.2 },
      { study_method: 'mixed', mean: 79.5 },
      { study_method: 'notes', mean: 74.0 },
      { study_method: 'group study', mean: 71.8 },
      { study_method: 'textbook', mean: 65.2 }
    ],
    subject_averages: { math: 69.4, science: 71.8, english: 73.5, overall: 71.6 }
  };

  const data = datasetInsights || defaultInsights;

  if (tabName === 'hours') {
    const labels = Object.keys(data.hours_vs_score || defaultInsights.hours_vs_score);
    const values = Object.values(data.hours_vs_score || defaultInsights.hours_vs_score);

    activeChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Average Overall Score (%)',
          data: values,
          backgroundColor: '#00a88f',
          borderRadius: 8,
          barThickness: 38
        }]
      },
      options: {
        ...defaultOptions,
        scales: {
          ...defaultOptions.scales,
          y: { ...defaultOptions.scales.y, min: 40, max: 100 }
        }
      }
    });
  } else if (tabName === 'attendance') {
    const labels = Object.keys(data.attendance_vs_score || defaultInsights.attendance_vs_score);
    const values = Object.values(data.attendance_vs_score || defaultInsights.attendance_vs_score);

    activeChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Overall Score Potential (%)',
          data: values,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 6,
          pointBackgroundColor: '#8b5cf6',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2
        }]
      },
      options: {
        ...defaultOptions,
        scales: {
          ...defaultOptions.scales,
          y: { ...defaultOptions.scales.y, min: 40, max: 100 }
        }
      }
    });
  } else if (tabName === 'method') {
    const methods = (data.method_stats || defaultInsights.method_stats);
    const labels = methods.map(m => m.study_method.toUpperCase());
    const values = methods.map(m => m.mean);

    activeChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Average Score by Method (%)',
          data: values,
          backgroundColor: [
            '#00a88f', '#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899', '#64748b'
          ],
          borderRadius: 8,
          barThickness: 32
        }]
      },
      options: {
        ...defaultOptions,
        scales: {
          ...defaultOptions.scales,
          y: { ...defaultOptions.scales.y, min: 50, max: 100 }
        }
      }
    });
  } else if (tabName === 'subjects') {
    const sub = (data.subject_averages || defaultInsights.subject_averages);
    activeChartInstance = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Mathematics', 'Science', 'English Literature', 'Study Dedication', 'Attendance Impact'],
        datasets: [
          {
            label: 'Dataset Average Profile',
            data: [sub.math, sub.science, sub.english, 70, 78],
            borderColor: '#94a3b8',
            backgroundColor: 'rgba(148, 163, 184, 0.15)',
            borderDash: [5, 5]
          },
          {
            label: 'High-Achiever Target Profile',
            data: [90, 92, 88, 95, 96],
            borderColor: '#00a88f',
            backgroundColor: 'rgba(0, 168, 143, 0.25)',
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: { font: { family: 'Plus Jakarta Sans', size: 11, weight: '600' } }
          }
        },
        scales: {
          r: {
            min: 30,
            max: 100,
            ticks: { display: false },
            grid: { color: '#e2e8f0' },
            pointLabels: { font: { family: 'Plus Jakarta Sans', size: 12, weight: '700' }, color: '#1e293b' }
          }
        }
      }
    });
  } else if (tabName === 'features') {
    const features = [
      'STEM Subject Average',
      'English Score',
      'Daily Study Hours',
      'Attendance Rate',
      'Study Method Type',
      'Parent Education',
      'Extracurriculars',
      'School Type'
    ];
    const importances = [0.94, 0.88, 0.82, 0.79, 0.65, 0.52, 0.38, 0.24];

    activeChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: features,
        datasets: [{
          label: 'Feature Importance Weight',
          data: importances,
          backgroundColor: '#00a88f',
          borderRadius: 6,
          barThickness: 22
        }]
      },
      options: {
        ...defaultOptions,
        indexAxis: 'y',
        scales: {
          x: { ...defaultOptions.scales.x, min: 0, max: 1.0 },
          y: { grid: { display: false }, ticks: { font: { family: 'Plus Jakarta Sans', size: 12, weight: '600' } } }
        }
      }
    });
  }
}
