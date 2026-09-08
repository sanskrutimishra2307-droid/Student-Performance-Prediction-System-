/**
 * Multi-Step Student Onboarding & Registration Flow
 * Collects necessary attributes and immediately generates a personalized AI dashboard.
 */

let currentStep = 1;

function initOnboarding() {
  currentStep = 1;
  showStep(currentStep);

  document.getElementById('btnNextStep1')?.addEventListener('click', () => {
    const name = document.getElementById('regName')?.value.trim();
    if (!name) {
      alert("Please enter the student's full name to continue.");
      return;
    }
    goToStep(2);
  });

  document.getElementById('btnBackStep2')?.addEventListener('click', () => goToStep(1));
  document.getElementById('btnNextStep2')?.addEventListener('click', () => goToStep(3));

  document.getElementById('btnBackStep3')?.addEventListener('click', () => goToStep(2));
  document.getElementById('btnFinishRegistration')?.addEventListener('click', finishRegistration);
}

function goToStep(step) {
  currentStep = step;
  showStep(currentStep);
}

function showStep(step) {
  // Hide all step sections
  document.getElementById('wizardSection1').style.display = (step === 1) ? 'block' : 'none';
  document.getElementById('wizardSection2').style.display = (step === 2) ? 'block' : 'none';
  document.getElementById('wizardSection3').style.display = (step === 3) ? 'block' : 'none';

  // Update step indicators
  for (let i = 1; i <= 3; i++) {
    const node = document.getElementById(`stepNode${i}`);
    if (!node) continue;
    node.classList.remove('active', 'completed');
    if (i === step) {
      node.classList.add('active');
    } else if (i < step) {
      node.classList.add('completed');
    }
  }
}

function finishRegistration() {
  const name = document.getElementById('regName')?.value.trim() || 'Alex Mercer';
  const age = parseInt(document.getElementById('regAge')?.value || 17);
  const gender = document.getElementById('regGender')?.value || 'female';
  const schoolType = document.getElementById('regSchoolType')?.value || 'public';
  const parentEd = document.getElementById('regParentEd')?.value || 'graduate';

  const studyHours = parseFloat(document.getElementById('regStudyHours')?.value || 4.0);
  const attendance = parseFloat(document.getElementById('regAttendance')?.value || 85.0);
  const internet = document.getElementById('regInternet')?.value || 'yes';
  const travel = document.getElementById('regTravel')?.value || '<15 min';
  const studyMethod = document.getElementById('regStudyMethod')?.value || 'online videos';
  const extraActivities = document.getElementById('regActivities')?.value || 'yes';

  const math = parseFloat(document.getElementById('regMath')?.value || 75.0);
  const science = parseFloat(document.getElementById('regScience')?.value || 80.0);
  const english = parseFloat(document.getElementById('regEnglish')?.value || 82.0);

  const overallAvg = (math + science + english) / 3;
  let gradeLetter = 'B+';
  if (overallAvg >= 90) gradeLetter = 'A+';
  else if (overallAvg >= 80) gradeLetter = 'A-';
  else if (overallAvg >= 70) gradeLetter = 'B';
  else if (overallAvg < 60) gradeLetter = 'C';

  const newStudentPersona = {
    id: 'student_' + Date.now(),
    name: name,
    role: 'Newly Enrolled',
    avatar: 'assets/avatar_emily.svg',
    grade: gradeLetter,
    gpa: `${gradeLetter} (${Math.round(overallAvg)}%)`,
    overall_score: Math.round(overallAvg * 10) / 10,
    data: {
      age: age,
      gender: gender,
      school_type: schoolType,
      parent_education: parentEd,
      study_hours: studyHours,
      attendance_percentage: attendance,
      internet_access: internet,
      travel_time: travel,
      extra_activities: extraActivities,
      study_method: studyMethod,
      math_score: math,
      science_score: science,
      english_score: english
    },
    homework_completed: 75,
    homework_total: 120,
    exams_completed: 4,
    exams_total: 12
  };

  // Add to active application personas list
  if (window.appState && window.appState.personas) {
    window.appState.personas.push(newStudentPersona);
    window.appState.activeStudent = newStudentPersona;
    window.renderStudentDashboard(newStudentPersona);
    window.switchTab('dashboard');
    window.showToast(`✨ Personalized AI Dashboard created for ${name}!`);
  }
}
