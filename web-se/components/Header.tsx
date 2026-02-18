export default function Header() {
  return (
    <footer className="bg-gradient-to-r from-kku-maroon to-red-900 text-white shadow-xl">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {/* <div className="w-12 h-12 bg-kku-gold rounded-full flex items-center justify-center">
              <span className="text-2xl font-bold text-kku-maroon">SEAI</span>
            </div> */}
            <div>
              <h1 className="text-2xl font-bold">Funny Fingers</h1>
              {/* <p className="text-sm text-gray-200">Object Detection Web Application</p> */}
            </div>
          </div>
          {/* <div className="hidden md:flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-lg">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-sm">API Connected</span>
          </div> */}
        </div>
      </div>
    </footer>
  )
}