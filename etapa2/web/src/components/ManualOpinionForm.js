import { useState } from 'react'
import {
  Button,
  Form,
  Col,
  ListGroup,
  InputGroup,
  Spinner,
} from 'react-bootstrap'
import { FaPlus, FaTrash, FaRedo, FaArrowLeft, FaSearch } from 'react-icons/fa'

const ManualOpinionForm = ({
  clearResults,
  loading,
  onBack,
  onSubmit,
  opinions,
  setOpinions,
}) => {
  const [textInput, setTextInput] = useState('')
  const [showOpinions, setShowOpinions] = useState(true)

  // Al cambiar la lista de opiniones, los resultados anteriores dejan de
  // corresponder con ella: hay mas (o menos) opiniones que predicciones.
  // Se descartan y se vuelve a la vista de edicion.
  const resetResults = () => {
    clearResults()
    setShowOpinions(true)
  }

  const addOpinion = () => {
    if (textInput.trim() !== '') {
      setOpinions([...opinions, textInput])
      setTextInput('')
      resetResults()
    }
  }

  const deleteOpinion = (index) => {
    setOpinions(opinions.filter((_, i) => i !== index))
    resetResults()
  }

  const handlePredict = () => {
    onSubmit()
    setShowOpinions(false)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      addOpinion()
    }
  }

  const handleRefresh = () => {
    setShowOpinions(true)
    setOpinions([])
    setTextInput('')
    clearResults()
  }

  return (
    <Col md={6} className='mx-auto'>
      <h4 className='text-center'>Agregar Opinión</h4>

      <InputGroup className='mb-3'>
        <Form.Control
          type='text'
          placeholder='Escribe tu opinión...'
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          onKeyUp={handleKeyPress}
        />
        <Button variant='primary' onClick={addOpinion}>
          <FaPlus />
        </Button>
      </InputGroup>

      {opinions.length > 0 && showOpinions && (
        <>
          <h4 className='text-center'>Opiniones Ingresadas</h4>
          <ListGroup className='mt-3'>
            {opinions.map((opinion, index) => (
              <ListGroup.Item
                key={index}
                className='d-flex justify-content-between align-items-center mb-2'
              >
                <span>{opinion}</span>
                <Button
                  variant='outline-danger'
                  size='sm'
                  onClick={() => deleteOpinion(index)}
                >
                  <FaTrash />
                </Button>
              </ListGroup.Item>
            ))}
          </ListGroup>

          <div className='text-center'>
            <Button className='mt-3' onClick={handlePredict} disabled={loading}>
              {loading ? (
                <>
                  <Spinner
                    as='span'
                    animation='border'
                    size='sm'
                    role='status'
                    aria-hidden='true'
                    className='me-2'
                  />
                  Cargando...
                </>
              ) : (
                <>
                  <FaSearch className='me-2' /> Predecir
                </>
              )}
            </Button>
          </div>
        </>
      )}

      <div className='text-center mt-4'>
        <Button className='me-2' variant='secondary' onClick={onBack}>
          <FaArrowLeft className='me-2' /> Volver
        </Button>
        {!showOpinions && (
          <Button variant='info' onClick={handleRefresh}>
            <FaRedo className='me-2' /> Refrescar
          </Button>
        )}
      </div>
    </Col>
  )
}

export default ManualOpinionForm
