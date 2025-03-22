let counter = 0;
let caloriesBurned = 0;
let interval;

// Function to start exercise tracking
function startExercise(exercise) {
  const weight = parseFloat(document.getElementById("weight").value);
  if (isNaN(weight)) {
    alert("Please enter your weight.");
    return;
  }

  // Reset counter and calories
  counter = 0;
  caloriesBurned = 0;
  updateUI();

  // Clear previous interval
  if (interval) clearInterval(interval);

  // Simulate exercise tracking
  interval = setInterval(() => {
    counter++;
    caloriesBurned = calculateCaloriesBurned(counter, weight, exercise);
    updateUI();
  }, 1000); // Update every second
}

// Function to calculate calories burned
function calculateCaloriesBurned(reps, weight, exercise) {
  let caloriesPerRep = 0;
  switch (exercise) {
    case "curl":
      caloriesPerRep = 0.25;
      break;
    case "situp":
      caloriesPerRep = 0.2;
      break;
    case "squat":
      caloriesPerRep = 0.3;
      break;
    case "lunge":
      caloriesPerRep = 0.22;
      break;
    default:
      caloriesPerRep = 0;
  }
  return reps * caloriesPerRep * (weight / 200);
}

// Function to update the UI
function updateUI() {
  document.getElementById("counter").textContent = counter;
  document.getElementById("calories").textContent = caloriesBurned.toFixed(2);
}

// Function to get diet recommendations
function getDietRecommendation() {
  const dietList = document.getElementById("diet-list");
  dietList.innerHTML = ""; // Clear previous recommendations

  // Example diet recommendations
  const recommendations = [
    "Brown Rice",
    "Chicken Breast",
    "Broccoli",
    "Sweet Potatoes",
    "Greek Yogurt",
  ];

  recommendations.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    dietList.appendChild(li);
  });
}