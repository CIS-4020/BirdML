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
                    <ImageUpload />
                    <div className="fileUpload">
                        <div className="image">
                            <img id="myImage" />
                        </div>
                        <div id="processBtn">WE UP IN HERE BIRDML'ing</div>
                    </div>
                </section>
            </div>
        </>
    )
}

export default App
