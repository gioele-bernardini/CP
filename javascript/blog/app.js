require('dotenv').config();

const express = require('express');

const app = express();  // Function to create app
const PORT = 5000 || process.env.PORT;

app.get('', (req, res) => {
  res.send('Hello, World!');
  // console.log('Request received!')
});

app.listen(PORT, () => {
  console.log(`App listening on port ${PORT}`);
})