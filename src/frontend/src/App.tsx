import './App.css'
import ImageUpload from './components/ImageUpload'

function App() {

    return (
        <>
            <div id="container">
                <div id="header">
                    <h1>BirdML</h1>
                </div>
                <section id="main">
                    <div>
                        <ImageUpload />
                    </div>
                    <div>
                        WE UP IN HERE BIRDML'ing
                        <img id="myImage" />
                    </div>
                </section>
            </div>
        </>
    )
}

export default App
