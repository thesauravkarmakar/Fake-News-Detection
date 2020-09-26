let covidInfo = (function () {
  let infoObj;

  //get India info
  fetch('https://covid19.mathdro.id/api/countries/ind')
    .then((response) => {
      return response.json();
    })
    .then((data) => {
      infoObj = data;
      setValues('india');
    });

  function setValues(country) {

    const countryId = country + '-info';
    const infoElm = document.getElementById(countryId);

    let info_confirmed_elm = infoElm.querySelector('.confirmed');

    let info_confirmed_value = infoObj.confirmed.value;
    info_confirmed_elm.querySelector('.value').innerText = info_confirmed_value;

    let info_recovered_elm = infoElm.querySelector('.recovered');
    let info_recovered_value = infoObj.recovered.value;
    info_recovered_elm.querySelector('.value').innerText = info_recovered_value;

    let info_deaths_elm = infoElm.querySelector('.deaths');
    let info_deaths_value = infoObj.deaths.value;
    info_deaths_elm.querySelector('.value').innerText = info_deaths_value;

    let info_active_elm = infoElm.querySelector('.active');
    info_active_elm.querySelector('.value').innerText = info_confirmed_value - (info_deaths_value + info_recovered_value);

    let info_updated_elm = infoElm.querySelector('.updated-date');
    info_updated_elm.querySelector('.detail').innerText = new Date(infoObj.lastUpdate).toLocaleDateString() + ", " + new Date(infoObj.lastUpdate).toLocaleTimeString();
  }
})();

