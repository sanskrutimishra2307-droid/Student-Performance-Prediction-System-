/**
 * Main Application Orchestrator for AURA EDTECH AI Dashboard
 */

window.appState = {
  activeTab: 'dashboard',
  activeStudent: null,
  personas: [],
  datasetInsights: null
};

// Initial Realistic Student Personas from Kaggle Dataset
const defaultPersonas = [
  {
    id: "adam",
    name: "Adam Chen",
    role: "High Achiever",
    avatar: "assets/avatar_adam.svg",
    grade: "A",
    gpa: "A (91%)",
    overall_score: 91.2,
    data: {
      age: 17,
      gender: "male",
      school_type: "public",
      parent_education: "post graduate",
      study_hours: 5.5,
      attendance_percentage: 94.0,
      internet_access: "yes",
      travel_time: "<15 min",
      extra_activities: "yes",
      study_method: "coaching",
      math_score: 88.0,
      science_score: 93.0,
      english_score: 91.0
    },
    homework_completed: 112,
    homework_total: 120,
    exams_completed: 11,
    exams_total: 12
  },
  {
    id: "michael",
    name: "Michael Brown",
    role: "Steady Performer",
    avatar: "assets/avatar_michael.svg",
    grade: "A-",
    gpa: "A- (84%)",
    overall_score: 84.5,
    data: {
      age: 16,
      gender: "male",
      school_type: "public",
      parent_education: "graduate",
      study_hours: 4.5,
      attendance_percentage: 82.0,
      internet_access: "yes",
      travel_time: "15-30 min",
      extra_activities: "yes",
      study_method: "notes",
      math_score: 68.0,
      science_score: 76.0,
      english_score: 82.0
    },
    homework_completed: 80,
    homework_total: 120,
    exams_completed: 5,
    exams_total: 12
  },
  {
    id: "sophia",
    name: "Sophia Martinez",
    role: "At-Risk Focus",
    avatar: "assets/avatar_sophia.svg",
    grade: "C-",
    gpa: "C (54%)",
    overall_score: 54.0,
    data: {
      age: 16,
      gender: "female",
      school_type: "public",
      parent_education: "high school",
      study_hours: 1.5,
      attendance_percentage: 62.0,
      internet_access: "no",
      travel_time: "30-60 min",
      extra_activities: "no",
      study_method: "textbook",
      math_score: 42.0,
      science_score: 51.0,
      english_score: 56.0
    },
    homework_completed: 45,
    homework_total: 120,
    exams_completed: 3,
    exams_total: 12
  },
  {
    id: "emily",
    name: "Emily Watson",
    role: "Rising Talent",
    avatar: "assets/avatar_emily.svg",
    grade: "B+",
    gpa: "B+ (78%)",
    overall_score: 78.4,
    data: {
      age: 17,
      gender: "female",
      school_type: "private",
      parent_education: "graduate",
      study_hours: 3.5,
      attendance_percentage: 86.0,
      internet_access: "yes",
      travel_time: "<15 min",
      extra_activities: "yes",
      study_method: "online videos",
      math_score: 74.0,
      science_score: 80.0,
      english_score: 81.0
    },
    homework_completed: 92,
    homework_total: 120,
    exams_completed: 8,
    exams_total: 12
  }
];

document.addEventListener('DOMContentLoaded', async () => {
  window.appState.personas = defaultPersonas;
  window.appState.activeStudent = defaultPersonas[1]; // Start with Michael Brown

  // Setup navigation tabs
  setupNavMenu();
  setupPersonaDropdown();
  setupExamsTabSwitcher();
  setupAnalysisTabs();

  // Initialize sub-engines
  initSimulator(window.appState.activeStudent.data);
  initOnboarding();

  // Fetch dataset insights if API available
  try {
    const res = await fetch('/api/v1/analytics/dataset-insights');
    if (res.ok) {
      window.appState.datasetInsights = await res.json();
    }
  } catch (e) {
    console.log("Using built-in dataset distributions.");
  }

  // Render initial student dashboard
  renderStudentDashboard(window.appState.activeStudent);
});

/**
 * Setup navigation sidebar menu
 */
function setupNavMenu() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = item.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

/**
 * Switch active visible tab
 */
window.switchTab = function(tabName) {
  window.appState.activeTab = tabName;

  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.toggle('active', item.getAttribute('data-tab') === tabName);
  });

  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.remove('active');
  });

  const targetView = document.getElementById(`view-${tabName}`);
  if (targetView) {
    targetView.classList.add('active');
  }

  if (tabName === 'performance') {
    renderAnalyticsChart('hours', window.appState.datasetInsights);
  } else if (tabName === 'simulator') {
    initSimulator(window.appState.activeStudent.data);
  }
};

/**
 * Setup Persona Switcher Dropdown
 */
function setupPersonaDropdown() {
  const pill = document.getElementById('studentProfilePill');
  const menu = document.getElementById('profileDropdownMenu');

  pill?.addEventListener('click', (e) => {
    e.stopPropagation();
    menu.classList.toggle('active');
  });

  document.addEventListener('click', () => {
    menu?.classList.remove('active');
  });

  renderPersonaMenuList();
}

function renderPersonaMenuList() {
  const menu = document.getElementById('profileDropdownMenu');
  if (!menu) return;

  menu.innerHTML = window.appState.personas.map(p => `
    <div class="persona-item ${p.id === window.appState.activeStudent?.id ? 'selected' : ''}" onclick="selectStudent('${p.id}')">
      <img src="${p.avatar}" style="width:24px;height:24px;border-radius:50%">
      <div>
        <div>${p.name}</div>
        <div style="font-size:10px;color:#64748b">${p.role} • ${p.grade}</div>
      </div>
    </div>
  `).join('') + `
    <div class="persona-item" style="border-top:1px solid #e2e8f0;margin-top:4px;color:#00a88f" onclick="switchTab('onboarding')">
      ➕ <strong>Register New Student</strong>
    </div>
  `;
}

window.selectStudent = function(studentId) {
  const student = window.appState.personas.find(p => p.id === studentId);
  if (student) {
    window.appState.activeStudent = student;
    renderStudentDashboard(student);
    initSimulator(student.data);
    showToast(`Switched to student profile: ${student.name}`);
  }
};

/**
 * Render Student Dashboard data matching reference UI
 */
window.renderStudentDashboard = function(student) {
  // 1. Update Profile Pill & Top Bar
  document.getElementById('activeStudentName').innerText = student.name.split(' ')[0];
  document.getElementById('activeStudentRole').innerText = student.role;
  document.getElementById('activeStudentAvatar').src = student.avatar;
  document.getElementById('topUserAvatar').src = student.avatar;
  document.getElementById('topUserName').innerText = student.name;
  document.getElementById('greetingTitle').innerText = `Welcome back, ${student.name.split(' ')[0]} 👋`;

  // 2. Hero Performance Card (Teal Card)
  document.getElementById('heroGpaGrade').innerText = student.grade;
  document.getElementById('heroHwCompleted').innerText = `${student.homework_completed}/${student.homework_total}`;
  document.getElementById('heroExamsCompleted').innerText = `${student.exams_completed}/${student.exams_total}`;
  
  // Set vertical capsules according to score level
  const score = student.overall_score;
  const vFillA = document.getElementById('vbarFillA');
  const vFillB = document.getElementById('vbarFillB');
  const vFillC = document.getElementById('vbarFillC');
  const vFillD = document.getElementById('vbarFillD');
  const vFillF = document.getElementById('vbarFillF');

  if (vFillA) vFillA.style.height = (score >= 85 ? '90%' : score >= 75 ? '50%' : '15%');
  if (vFillB) vFillB.style.height = (score >= 70 && score < 85 ? '85%' : '35%');
  if (vFillC) vFillC.style.height = (score >= 55 && score < 70 ? '80%' : '20%');
  if (vFillD) vFillD.style.height = (score < 55 ? '60%' : '8%');
  if (vFillF) vFillF.style.height = (score < 45 ? '75%' : '4%');

  // 3. AI Promo / Highlight Card (Powered by Real ML Inference)
  const promoTitle = document.getElementById('heroPromoTitle');
  if (promoTitle) {
    fetch('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(student.data)
    })
    .then(res => res.json())
    .then(mlRes => {
      if (mlRes.prediction) {
        promoTitle.innerHTML = `ML Forecast: <strong>${mlRes.prediction.toUpperCase()}</strong> (${(mlRes.confidence * 100).toFixed(0)}% Conf.)<br><span style="font-size:11px;opacity:0.9">${mlRes.explanation?.summary_statement?.slice(0, 55) || 'Model evaluated'}...</span>`;
      }
    })
    .catch(() => {
      promoTitle.innerText = "Targeting +8% GPA with Calculus Coaching!";
    });
  }

  // 4. Update Course Cards with student-specific scores
  const mathGpa = document.getElementById('course1Gpa');
  const mathProgress = document.getElementById('course1Progress');
  const mathPct = document.getElementById('course1Pct');

  const sciGpa = document.getElementById('course2Gpa');
  const sciProgress = document.getElementById('course2Progress');
  const sciPct = document.getElementById('course2Pct');

  if (mathGpa) mathGpa.innerText = `Score: ${student.data.math_score || 70} pts`;
  if (mathProgress) mathProgress.style.width = `${Math.min(100, (student.data.math_score || 70))}%`;
  if (mathPct) mathPct.innerText = `${Math.min(100, (student.data.math_score || 70))}%`;

  if (sciGpa) sciGpa.innerText = `Score: ${student.data.science_score || 75} pts`;
  if (sciProgress) sciProgress.style.width = `${Math.min(100, (student.data.science_score || 75))}%`;
  if (sciPct) sciPct.innerText = `${Math.min(100, (student.data.science_score || 75))}%`;

  renderPersonaMenuList();
};

/**
 * Setup Exams / Homeworks Tab Toggle in Right Column
 */
function setupExamsTabSwitcher() {
  const btnExams = document.getElementById('tabBtnExams');
  const btnHomeworks = document.getElementById('tabBtnHomeworks');
  const list = document.getElementById('upcomingItemsList');

  btnExams?.addEventListener('click', () => {
    btnExams.classList.add('active');
    btnHomeworks?.classList.remove('active');
    renderUpcomingItems('exams');
  });

  btnHomeworks?.addEventListener('click', () => {
    btnHomeworks.classList.add('active');
    btnExams?.classList.remove('active');
    renderUpcomingItems('homeworks');
  });
}

function renderUpcomingItems(type) {
  const list = document.getElementById('upcomingItemsList');
  if (!list) return;

  if (type === 'exams') {
    list.innerHTML = `
      <div class="upcoming-item-card">
        <div class="date-badge">
          <div class="date-badge-day">10</div>
          <div class="date-badge-month">August</div>
        </div>
        <div class="upcoming-item-details">
          <div class="upcoming-item-title">Calculus & Linear Algebra Final</div>
          <div class="upcoming-item-desc">Comprehensive exam covering differential equations & matrices.</div>
        </div>
        <div class="upcoming-item-time">
          03:00 PM
          <img src="assets/avatar_instructor1.svg">
        </div>
      </div>
      <div class="upcoming-item-card">
        <div class="date-badge">
          <div class="date-badge-day">8</div>
          <div class="date-badge-month">June</div>
        </div>
        <div class="upcoming-item-details">
          <div class="upcoming-item-title">Experimental Physics & Mechanics</div>
          <div class="upcoming-item-desc">Lab assessment & theoretical thermodynamics.</div>
        </div>
        <div class="upcoming-item-time">
          11:00 AM
          <img src="assets/avatar_instructor2.svg">
        </div>
      </div>
      <div class="upcoming-item-card">
        <div class="date-badge alert">
          <div class="date-badge-day">2</div>
          <div class="date-badge-month">June</div>
        </div>
        <div class="upcoming-item-details">
          <div class="upcoming-item-title">Creative Technical Writing</div>
          <div class="upcoming-item-desc">Project presentation & abstract defense.</div>
        </div>
        <div class="upcoming-item-time">
          <span style="color:#ef4444;font-size:10px;font-weight:800">RETAKE</span>
          <img src="assets/avatar_instructor1.svg">
        </div>
      </div>
    `;
  } else {
    list.innerHTML = `
      <div class="upcoming-item-card">
        <div class="date-badge">
          <div class="date-badge-day">14</div>
          <div class="date-badge-month">August</div>
        </div>
        <div class="upcoming-item-details">
          <div class="upcoming-item-title">Problem Set #6: Vector Spaces</div>
          <div class="upcoming-item-desc">Mathematics • 12 Problems due by midnight.</div>
        </div>
        <div class="upcoming-item-time">
          11:59 PM
          <img src="assets/avatar_instructor1.svg">
        </div>
      </div>
      <div class="upcoming-item-card">
        <div class="date-badge">
          <div class="date-badge-day">18</div>
          <div class="date-badge-month">August</div>
        </div>
        <div class="upcoming-item-details">
          <div class="upcoming-item-title">Physics Lab Report: Optics</div>
          <div class="upcoming-item-desc">Science • Submit simulation charts and conclusion.</div>
        </div>
        <div class="upcoming-item-time">
          05:00 PM
          <img src="assets/avatar_instructor2.svg">
        </div>
      </div>
    `;
  }
}

/**
 * Setup Analysis Tabs (Study Hours, Attendance, Study Method, Subjects, Features)
 */
function setupAnalysisTabs() {
  document.querySelectorAll('.analysis-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.analysis-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tabName = btn.getAttribute('data-analysis-tab');
      renderAnalyticsChart(tabName, window.appState.datasetInsights);
    });
  });
}

/**
 * Toast Notification Utility
 */
window.showToast = function(message) {
  let toast = document.getElementById('toastNotification');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toastNotification';
    toast.className = 'toast-notification';
    document.body.appendChild(toast);
  }

  toast.innerHTML = `<span>💡</span> ${message}`;
  toast.classList.add('show');

  setTimeout(() => {
    toast.classList.remove('show');
  }, 3500);
};

/**
 * Export Student Intelligence Report (PDF / Printable view)
 */
window.exportStudentReport = function() {
  const student = window.appState.activeStudent;
  window.showToast(`📄 Generating Academic Dossier for ${student.name}...`);
  setTimeout(() => {
    window.print();
  }, 800);
};
