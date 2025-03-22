const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());

const dietRecommendations = [
  "Brown Rice",
  "Chicken Breast",
  "Broccoli",
  "Sweet Potatoes",
  "Greek Yogurt",
];

app.get("/diet", (req, res) => {
  res.json(dietRecommendations);
});

app.listen(5000, () => {
  console.log("Backend running on http://localhost:5000");
});