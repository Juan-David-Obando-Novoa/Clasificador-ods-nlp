import { useState } from 'react'
import { ListGroup, ProgressBar, Pagination, Button } from 'react-bootstrap'

const ODS_DESCRIPTIONS = {
  3: 'Salud y bienestar',
  4: 'Educación de calidad',
  5: 'Igualdad de género',
}

const ITEMS_PER_PAGE_OPTIONS = [10, 15, 20, 50, 100]

const PredictionResults = ({ results, opinions }) => {
  const colors = ['#4c9f38', '#c5192d', '#ff3a21']

  const [itemsPerPage, setItemsPerPage] = useState(10)
  const [currentPage, setCurrentPage] = useState(1)
  const [expandedOpinions, setExpandedOpinions] = useState({})

  const predictions = results?.predictions ?? []
  const probabilities = results?.probabilities ?? []
  // Las clases las informa la API; antes estaban fijas como i + 3.
  const classes = results?.classes ?? [3, 4, 5]

  // Solo se muestran las opiniones que tienen prediccion. Si la lista cambio
  // despues de predecir, no se intenta leer un indice que no existe.
  const shownOpinions = opinions.slice(0, predictions.length)
  const totalPages = Math.ceil(shownOpinions.length / itemsPerPage)

  const handlePageChange = (page) => {
    setCurrentPage(page)
  }

  const startIndex = (currentPage - 1) * itemsPerPage
  const selectedOpinions = shownOpinions.slice(
    startIndex,
    startIndex + itemsPerPage
  )
  const selectedPredictions = predictions.slice(
    startIndex,
    startIndex + itemsPerPage
  )
  const selectedProbabilities = probabilities.slice(
    startIndex,
    startIndex + itemsPerPage
  )

  const pageLimit = 11
  const pagesToShow = []
  let firstPage = Math.max(1, currentPage - Math.floor(pageLimit / 2))
  let lastPage = Math.min(totalPages, firstPage + pageLimit - 1)

  if (lastPage - firstPage < pageLimit - 1) {
    firstPage = Math.max(1, lastPage - pageLimit + 1)
  }

  for (let i = firstPage; i <= lastPage; i++) {
    pagesToShow.push(i)
  }

  const toggleExpansion = (index) => {
    setExpandedOpinions((prev) => ({
      ...prev,
      [index]: !prev[index],
    }))
  }

  return (
    <>
      <div className='mb-3'>
        <label htmlFor='itemsPerPage' className='me-2'>
          Opiniones por página:
        </label>
        <select
          id='itemsPerPage'
          value={itemsPerPage}
          onChange={(e) => {
            setItemsPerPage(+e.target.value)
            setCurrentPage(1)
            setExpandedOpinions({})
          }}
        >
          {ITEMS_PER_PAGE_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>

      <ListGroup>
        {selectedOpinions.map((opinion, index) => {
          const opinionWords = opinion.split(' ')

          const displayOpinion =
            expandedOpinions[index] || opinionWords.length <= 50
              ? opinion
              : opinionWords.slice(0, 50).join(' ') + '...'

          return (
            <ListGroup.Item
              key={index}
              className='d-flex align-items-center'
              style={{ gap: '20px' }}
            >
              <div style={{ flex: '0 0 40%', overflow: 'hidden' }}>
                <span>{displayOpinion}</span>
                {opinionWords.length > 50 && (
                  <Button
                    variant='link'
                    onClick={() => toggleExpansion(index)}
                    className='p-0 ms-2'
                  >
                    {expandedOpinions[index] ? 'Ver menos' : 'Ver más'}
                  </Button>
                )}
              </div>
              <div
                style={{
                  flex: '0 0 20%',
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <strong className='me-2'>
                  ODS {selectedPredictions[index]}:
                </strong>
                <span>{ODS_DESCRIPTIONS[selectedPredictions[index]]}</span>
              </div>
              <div style={{ flex: '1' }}>
                <ProgressBar style={{ height: '30px' }}>
                  {(selectedProbabilities[index] ?? []).map((prob, i) => (
                    <ProgressBar
                      now={prob * 100}
                      key={i}
                      style={{
                        backgroundColor: colors[i],
                        height: '30px',
                        fontWeight: 'bold',
                      }}
                      title={`ODS ${classes[i]}: ${(prob * 100).toFixed(2)}%`}
                      label={`${classes[i]}`}
                    >
                      <span style={{ color: 'white', fontWeight: 'bold' }}>
                        {classes[i]}
                      </span>
                    </ProgressBar>
                  ))}
                </ProgressBar>
              </div>
            </ListGroup.Item>
          )
        })}
      </ListGroup>

      <div className='d-flex justify-content-center mt-4'>
        <Pagination>
          <Pagination.Prev
            onClick={() =>
              handlePageChange(currentPage > 1 ? currentPage - 1 : totalPages)
            }
            disabled={totalPages === 0}
          />
          {pagesToShow.map((page) => (
            <Pagination.Item
              key={page}
              active={page === currentPage}
              onClick={() => handlePageChange(page)}
            >
              {page}
            </Pagination.Item>
          ))}
          <Pagination.Next
            onClick={() =>
              handlePageChange(currentPage < totalPages ? currentPage + 1 : 1)
            }
            disabled={totalPages === 0}
          />
        </Pagination>
      </div>
    </>
  )
}

export default PredictionResults
